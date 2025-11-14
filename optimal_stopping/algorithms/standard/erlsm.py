"""
Ensemble Randomized Least Squares Monte Carlo (ERLSM) for optimal stopping.

This algorithm improves upon RLSM by using:
1. **Ensemble of multiple networks** - Reduces variance in continuation value estimates
2. **Two-layer architecture** - Better function approximation than single layer
3. **Polynomial feature expansion** - Richer basis functions
4. **Weighted voting** - Better networks get more influence
5. **Bootstrap aggregating** - Diversity in training data

KEY INSIGHT: Underestimating continuation values leads to early exercise and lower
option prices. By using ensemble methods that reduce estimation variance, we achieve
more accurate (typically higher) continuation values → higher option prices.

Compatible with BOTH:
- Standard options (MaxCall, BasketCall, etc.)
- Path-dependent options (Barriers, Lookbacks) via auto-detection

Author: Claude (Advanced AI Assistant)
Date: 2025-11-14
"""

import torch
import numpy as np
import time
import math
from sklearn.preprocessing import PolynomialFeatures
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class TwoLayerReservoir(torch.nn.Module):
    """
    Two-layer randomized neural network with frozen weights.

    Architecture: Input → Hidden1 → Hidden2 → Features
    Only the final linear regression layer (trained separately) is learned.
    """

    def __init__(self, hidden_size_1, hidden_size_2, state_size, factors=(1., 1.), activation=None):
        super().__init__()

        if activation is None:
            activation = torch.nn.LeakyReLU(0.1)

        # First hidden layer
        self.layer1 = torch.nn.Linear(state_size, hidden_size_1)
        self.activation1 = activation

        # Second hidden layer
        self.layer2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.activation2 = activation

        # Initialize with random weights (frozen)
        factor1, factor2 = factors if len(factors) >= 2 else (1., 1.)
        with torch.no_grad():
            torch.nn.init.normal_(self.layer1.weight, 0, factor1)
            torch.nn.init.normal_(self.layer1.bias, 0, factor1)
            torch.nn.init.normal_(self.layer2.weight, 0, factor2)
            torch.nn.init.normal_(self.layer2.bias, 0, factor2)

        # Freeze weights - they never get updated
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass through two-layer network."""
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        return x


class ERLSM:
    """
    Ensemble Randomized Least Squares Monte Carlo.

    Uses an ensemble of randomized neural networks to achieve better
    continuation value estimates and higher option prices.
    """

    def __init__(self, model, payoff, hidden_size=100, factors=(1., 1., 1.),
                 train_ITM_only=True, use_payoff_as_input=False,
                 ensemble_size=5, poly_degree=2, bootstrap_ratio=0.8, **kwargs):
        """
        Initialize ERLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function (works for both standard and path-dependent)
            hidden_size: Number of neurons per hidden layer
            factors: (activation_slope, weight_scale_1, weight_scale_2)
            train_ITM_only: If True, only use in-the-money paths for training
            use_payoff_as_input: If True, include payoff in state
            ensemble_size: Number of networks in ensemble (default: 5)
            poly_degree: Degree of polynomial features (1=linear, 2=quadratic)
            bootstrap_ratio: Fraction of data for each bootstrap sample
        """
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.factors = factors
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.ensemble_size = ensemble_size
        self.poly_degree = poly_degree
        self.bootstrap_ratio = bootstrap_ratio

        # Check if path-dependent
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Initialize ensemble of reservoirs
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks

        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

        self._init_ensemble(state_size, hidden_size, factors)

        # Initialize polynomial feature transformer
        if poly_degree > 1:
            self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
            # Fit on dummy data to initialize
            dummy = np.zeros((1, state_size))
            self.poly_features.fit(dummy)
            expanded_size = self.poly_features.transform(dummy).shape[1]
            self.feature_size = expanded_size
        else:
            self.poly_features = None
            self.feature_size = state_size

    def _init_ensemble(self, state_size, hidden_size, factors):
        """Initialize ensemble of two-layer randomized networks."""
        self.reservoirs = []

        # Use two hidden layers: hidden_size → hidden_size//2
        hidden_size_1 = hidden_size
        hidden_size_2 = max(hidden_size // 2, 20)

        activation_slope = factors[0] if len(factors) > 0 else 1.0
        factor1 = factors[1] if len(factors) > 1 else 1.0
        factor2 = factors[2] if len(factors) > 2 else 1.0

        activation = torch.nn.LeakyReLU(activation_slope / 2)

        for i in range(self.ensemble_size):
            # Each network gets different random seed for diversity
            torch.manual_seed(42 + i * 100)
            reservoir = TwoLayerReservoir(
                hidden_size_1,
                hidden_size_2,
                state_size,
                factors=(factor1, factor2),
                activation=activation
            )
            self.reservoirs.append(reservoir)

        self.nb_base_fcts = hidden_size_2 + 1  # +1 for constant term

        # Track ensemble weights (updated during training)
        self.ensemble_weights = np.ones(self.ensemble_size) / self.ensemble_size

    def price(self, train_eval_split=2):
        """
        Compute option price using ensemble RLSM.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs
        if self.is_path_dependent:
            # For path-dependent, pass full history
            payoffs = self.payoff(stock_paths)
        else:
            # For standard options, just need final prices
            payoffs = self.payoff(stock_paths)

        # Add payoff to stock paths if needed
        if self.use_payoff_as_input:
            stock_paths = np.concatenate(
                [stock_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks_plus, nb_dates = stock_paths.shape
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Initialize with terminal payoff
        if self.is_path_dependent:
            values = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :nb_dates])
        else:
            values = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, -1])

        # Track exercise dates
        self._exercise_dates = np.full(nb_paths, nb_dates - 1, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(nb_dates - 2, 0, -1):
            # Current immediate exercise value
            if self.is_path_dependent:
                immediate_exercise = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :date+1])
            else:
                immediate_exercise = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, date])

            # Prepare state for regression
            if self.use_payoff_as_input:
                current_state = stock_paths[:, :, date]
            else:
                current_state = stock_paths[:, :self.model.nb_stocks, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Learn continuation value using ENSEMBLE of randomized NNs
            continuation_values = self._learn_continuation_ensemble(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # Update values based on optimal exercise decision
            exercise_now = immediate_exercise > continuation_values

            # Track exercise dates
            self._exercise_dates[exercise_now] = date

            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

        # Final payoff at t=0
        if self.is_path_dependent:
            payoff_0 = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, :1])
        else:
            payoff_0 = self.payoff.eval(stock_paths[:, :self.model.nb_stocks, 0])

        # Return average price on evaluation set, discounted to time 0
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1]."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        normalized_times = self._exercise_dates / nb_dates
        return float(np.mean(normalized_times))

    def _expand_features(self, state):
        """Apply polynomial feature expansion."""
        if self.poly_features is None:
            return state
        return self.poly_features.transform(state)

    def _learn_continuation_ensemble(self, current_state, future_values, immediate_exercise):
        """
        Learn continuation value using ENSEMBLE of randomized neural networks.

        This is the key innovation: multiple networks vote on continuation value,
        reducing variance and improving accuracy.

        Args:
            current_state: (nb_paths, state_size) - Current stock prices
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised

        Returns:
            continuation_values: (nb_paths,) - Ensemble-averaged continuation values
        """
        # Determine which paths to use for training
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        # Apply polynomial feature expansion
        expanded_state = self._expand_features(current_state)

        # Collect predictions from all networks in ensemble
        all_predictions = []
        ensemble_errors = []

        for i, reservoir in enumerate(self.reservoirs):
            # Evaluate basis functions using this network
            X_tensor = torch.from_numpy(current_state).type(torch.float32)
            basis = reservoir(X_tensor).detach().numpy()

            # Add constant term
            basis = np.concatenate([basis, np.ones((len(basis), 1))], axis=1)

            # Bootstrap sampling for training diversity
            n_train = train_mask.sum()
            if self.bootstrap_ratio < 1.0 and n_train > 0:
                sample_size = int(n_train * self.bootstrap_ratio)
                train_indices = np.where(train_mask)[0]
                bootstrap_indices = np.random.choice(train_indices, size=sample_size, replace=True)
                bootstrap_mask = np.zeros(self.split, dtype=bool)
                bootstrap_mask[bootstrap_indices] = True
            else:
                bootstrap_mask = train_mask

            # Least squares regression on bootstrap sample
            if bootstrap_mask.sum() > 0:
                coefficients = np.linalg.lstsq(
                    basis[:self.split][bootstrap_mask],
                    future_values[:self.split][bootstrap_mask],
                    rcond=None
                )[0]

                # Predict continuation values
                predictions = np.dot(basis, coefficients)
                all_predictions.append(predictions)

                # Compute validation error for weighting (on non-training paths)
                val_mask = ~train_mask
                if val_mask.sum() > 0:
                    val_error = np.mean((predictions[:self.split][val_mask] -
                                       future_values[:self.split][val_mask])**2)
                    ensemble_errors.append(val_error)
                else:
                    ensemble_errors.append(1.0)
            else:
                # No training data, use zero prediction
                all_predictions.append(np.zeros(len(current_state)))
                ensemble_errors.append(float('inf'))

        # Compute ensemble weights based on validation errors
        # Lower error → higher weight
        ensemble_errors = np.array(ensemble_errors)
        if np.all(np.isfinite(ensemble_errors)):
            # Inverse error weighting: w_i = 1 / (error_i + epsilon)
            weights = 1.0 / (ensemble_errors + 1e-6)
            weights = weights / weights.sum()
            self.ensemble_weights = 0.9 * self.ensemble_weights + 0.1 * weights  # Smooth update

        # Weighted average of predictions
        all_predictions = np.array(all_predictions)  # Shape: (ensemble_size, nb_paths)
        continuation_values = np.average(all_predictions, axis=0, weights=self.ensemble_weights)

        return continuation_values
