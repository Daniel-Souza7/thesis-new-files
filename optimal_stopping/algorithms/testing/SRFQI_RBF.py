import numpy as np
import torch
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class SRFQI_RBF:
    """
    Specialized RFQI with Time-Localized RBFs (SRFQI_RBF).

    UNIVERSAL SOLVER:
    - Handles Path-Dependent Options (Barriers, Lookbacks)
    - Handles Standard Options (Calls, Puts, Baskets)

    KEY INNOVATION:
    Replaces simple 'time' input with a vector of Radial Basis Functions (RBFs).
    This forces the global neural network to learn 'local' time dynamics,
    fixing the underfitting issue seen in standard SRFQI for barrier options.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=100,
                 nb_time_buckets=20, factors=(1.,), train_ITM_only=True,
                 use_payoff_as_input=False, **kwargs):
        """
        Args:
            model: Stock model
            payoff: Payoff function (Standard OR Path-Dependent)
            nb_epochs: Number of training epochs (10-20 is usually enough now)
            hidden_size: Neurons in reservoir (100 is usually sufficient with RBFs)
            nb_time_buckets: Number of Gaussian time centers (e.g., 20)
            factors: (activation_slope, weight_scale)
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.nb_time_buckets = nb_time_buckets

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Detect if option is path dependent
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

        # Initialize randomized neural network
        # SRFQI_RBF uses fixed hidden_size=100 as default
        if hidden_size is None:
            hidden_size = 100
        if hidden_size < 0:
            hidden_size = max(model.nb_stocks * abs(hidden_size), 5)

        self.dim_out = hidden_size
        self.nb_base_fcts = self.dim_out + 1

        # STATE SIZE CALCULATION:
        # Stocks + Variance(opt) + Time_RBFs + Payoff(opt)
        # Note: We replaced the 2 standard time features (t, 1-t) with nb_time_buckets
        self.state_size = (model.nb_stocks * (1 + self.use_var) +
                           self.nb_time_buckets +
                           self.use_payoff_as_input * 1)

        self._init_reservoir(factors)

    def _init_reservoir(self, factors):
        """Initialize randomized neural network."""
        self.reservoir2 = randomized_neural_networks.Reservoir2(
            self.dim_out,
            self.state_size,
            factors=factors[1:] if len(factors) > 1 else (),
            activation=torch.nn.LeakyReLU(factors[0] / 2)
        )

    def evaluate_bases_all(self, stock_price):
        """
        Evaluate basis functions using Time-Localized RBFs.

        Instead of passing 't', we pass a vector of activations that
        peak at specific times. This acts like a 'soft switch', allowing
        the network to use different weights for different times.
        """
        stocks = torch.from_numpy(stock_price).type(torch.float32)
        stocks = stocks.permute(0, 2, 1)  # (nb_paths, nb_dates+1, nb_stocks)

        nb_paths, nb_dates, _ = stocks.shape

        # --- TIME LOCALIZATION STRATEGY (RBFs) ---
        # Create a time tensor [0.0, ..., 1.0] repeated for batch
        time_grid = torch.linspace(0, 1, nb_dates)
        time_tensor = time_grid.view(1, -1, 1).repeat(nb_paths, 1, 1)  # (batch, dates, 1)

        # Create Centers for the Gaussians
        centers = torch.linspace(0, 1, self.nb_time_buckets)

        # Bandwidth (Gamma): Controls how 'wide' the bell curve is.
        # Heuristic: N^2 ensures curves overlap slightly but remain distinct.
        gamma = self.nb_time_buckets ** 2

        # Generate RBF Features
        # exp(-gamma * (t - center)^2)
        rbf_list = []
        for c in centers:
            rbf = torch.exp(-gamma * (time_tensor - c) ** 2)
            rbf_list.append(rbf)

        time_features = torch.cat(rbf_list, dim=-1)  # (batch, dates, nb_time_buckets)

        # Concatenate Stocks + Time RBFs
        # Note: We do NOT add raw time/1-time anymore. RBFs cover it better.
        inputs = torch.cat([stocks, time_features], dim=-1)

        # Evaluate reservoir
        random_base = self.reservoir2(inputs)

        # Add constant term (Bias)
        random_base = torch.cat([
            random_base,
            torch.ones([nb_paths, nb_dates, 1])
        ], dim=-1)

        return random_base.detach().numpy()

    def _compute_payoffs(self, stock_paths):
        """
        Smartly computes payoffs based on option type.
        """
        if self.is_path_dependent:
            # Path Dependent: Must loop and pass history
            nb_paths, nb_stocks, nb_dates_plus_one = stock_paths.shape
            payoffs = np.zeros((nb_paths, nb_dates_plus_one))

            # Use a sliding window of history
            for date in range(nb_dates_plus_one):
                path_history = stock_paths[:, :, :date + 1]
                payoffs[:, date] = self.payoff.eval(path_history)
            return payoffs
        else:
            # Standard Option: Vectorized call
            return self.payoff(stock_paths)

    def price(self, train_eval_split=2):
        """
        Compute option price.
        """
        t_start = time.time()

        # 1. Generate Paths
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

        # 2. Compute Payoffs (Universal Handler)
        payoffs = self._compute_payoffs(stock_paths)

        # 3. Prepare Inputs for Neural Network
        input_paths = stock_paths.copy()

        if self.use_payoff_as_input:
            input_paths = np.concatenate(
                [input_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )

        if self.use_var and var_paths is not None:
            input_paths = np.concatenate([input_paths, var_paths], axis=1)

        # Split Training/Evaluation
        self.split = len(stock_paths) // train_eval_split

        # 4. Evaluate Basis Functions (Uses RBFs now)
        eval_bases = self.evaluate_bases_all(input_paths)

        # Constants
        nb_dates = self.model.nb_dates
        deltaT = self.model.maturity / nb_dates
        discount_factor = math.exp(-self.model.rate * deltaT)
        self.nb_base_fcts = eval_bases.shape[2]

        # 5. Fitted Q-Iteration (Training)
        weights = np.zeros(self.nb_base_fcts, dtype=float)

        for epoch in range(self.nb_epochs):
            # Continuation value (prediction at t+1)
            continuation_value = np.dot(eval_bases[:self.split, 1:, :], weights)
            continuation_value = np.maximum(0, continuation_value)

            # Target: Max of Immediate Payoff vs Continuation
            indicator_stop = np.maximum(payoffs[:self.split, 1:], continuation_value)

            # Regression Matrix U (Sum over Paths AND Time)
            matrixU = np.tensordot(
                eval_bases[:self.split, :-1, :],
                eval_bases[:self.split, :-1, :],
                axes=([0, 1], [0, 1])
            )

            # Regression Vector V
            vectorV = np.sum(
                eval_bases[:self.split, :-1, :] * discount_factor * np.repeat(
                    np.expand_dims(indicator_stop, axis=2),
                    self.nb_base_fcts,
                    axis=2
                ),
                axis=(0, 1)
            )

            # Solve Linear System
            weights = np.linalg.solve(matrixU, vectorV)

        self.weights = weights

        # 6. Final Evaluation
        continuation_value = np.dot(eval_bases, weights)
        continuation_value = np.maximum(0, continuation_value)

        # Exercise Decision
        which = (payoffs > continuation_value) * 1
        which[:, -1] = 1  # Must exercise at T
        which[:, 0] = 0  # Cannot exercise at 0

        # Find first exercise date
        ex_dates = np.argmax(which, axis=1)
        self._exercise_dates = ex_dates.copy()

        # Calculate Price
        prices = np.take_along_axis(
            payoffs,
            np.expand_dims(ex_dates, axis=1),
            axis=1
        ).reshape(-1) * (discount_factor ** ex_dates)

        # Handle max with initial payoff (American option condition at t=0)
        price = max(np.mean(prices[self.split:]), payoffs[0, 0])

        return price, time_path_gen

    def get_exercise_time(self):
        """Return average exercise time normalized."""
        if not hasattr(self, '_exercise_dates'):
            return None
        nb_dates = self.model.nb_dates
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        For video/visualization: Apply learned policy to new paths.
        """
        if not hasattr(self, 'weights'):
            raise ValueError("Train model with price() first.")

        # Compute payoffs
        payoffs = self._compute_payoffs(stock_paths)

        # Prepare inputs
        input_paths = stock_paths.copy()
        if self.use_payoff_as_input:
            input_paths = np.concatenate(
                [input_paths, np.expand_dims(payoffs, axis=1)], axis=1
            )
        if self.use_var and var_paths is not None:
            input_paths = np.concatenate([input_paths, var_paths], axis=1)

        # Evaluate bases
        eval_bases = self.evaluate_bases_all(input_paths)

        # Standard Backward Induction using learned weights
        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        values = payoffs[:, -1].copy()
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        # Pre-compute all continuation values to be fast
        all_continuation_values = np.dot(eval_bases, self.weights)

        for date in range(nb_dates - 1, 0, -1):
            immediate_exercise = payoffs[:, date]
            cont_value = np.maximum(0, all_continuation_values[:, date])

            exercise_now = immediate_exercise > cont_value
            values = np.where(exercise_now, immediate_exercise, values * disc_factor)
            exercise_dates[exercise_now] = date

        payoff_values = np.array([payoffs[i, exercise_dates[i]] for i in range(nb_paths)])
        price = np.mean(values)

        return exercise_dates, payoff_values, price