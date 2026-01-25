"""
Neural Least Squares Monte Carlo (NLSM) for American option pricing.

Simple benchmark implementation of the NLSM algorithm from:
"Neural network regression for Bermudan option pricing"
(Lapeyre and Lelong, 2019)
"""

import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import neural_networks


def init_weights(m):
    """Initialize neural network weights."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class NeuralNetworkPricer:
    """
    Computes American option price by Neural Least Squares Monte Carlo (NLSM).

    Uses a neural network to approximate the continuation value function.
    """

    def __init__(self, model, payoff, nb_epochs=20, hidden_size=10,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False):
        """
        Initialize NLSM pricer.

        Args:
            model: Stock model
            payoff: Payoff function
            nb_epochs: Number of training epochs per time step
            hidden_size: Number of neurons in hidden layer
            factors: Ignored (for API compatibility)
            train_ITM_only: If True, only train on in-the-money paths
            use_payoff_as_input: If True, include payoff as feature
        """
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        # NLSM uses fixed hidden_size=10 as default
        if hidden_size is None:
            hidden_size = 10
        self.hidden_size = hidden_size
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths
        self.use_var = getattr(model, 'return_var', False)

        # Create neural network once (reference methodology for speed)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1
        self.state_size = state_size
        self.neural_network = neural_networks.NetworkNLSM(
            self.state_size, hidden_size=self.hidden_size
        ).double()

        # Note: No network storage for performance (would need 2-5x more time)
        # backward_induction_on_paths() not supported with this approach
        self._learned_networks = {}

    def price(self, train_eval_split=2):
        """
        Compute option price using NLSM backward induction.

        Args:
            train_eval_split: Ratio for splitting paths into training/evaluation

        Returns:
            tuple: (price, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
            torch.manual_seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs for all paths
        payoffs = self.payoff(stock_paths)

        # Discount all payoffs upfront (reference methodology)
        power = np.arange(0, self.model.nb_dates + 1)
        disc_factors = np.exp(
            (-self.model.rate) * self.model.maturity / self.model.nb_dates * power
        )
        disc_factors = np.repeat(
            np.expand_dims(disc_factors, axis=0), repeats=payoffs.shape[0], axis=0
        )
        payoffs = payoffs * disc_factors

        # Split into training and evaluation sets
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape

        # Initialize with terminal payoff (already discounted)
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates, not nb_dates-1)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Backward induction from T-1 to 1
        for date in range(self.model.nb_dates - 1, 0, -1):
            # Current immediate exercise value (already discounted)
            immediate_exercise = payoffs[:, date]

            # Prepare state for regression
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Learn continuation value using neural network
            # Note: values already contains discounted future payoffs
            continuation_values = self._learn_continuation(
                current_state, values, immediate_exercise, date
            )

            # Update values: max(exercise now, continue)
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            # Don't discount values[~exercise_now] - already discounted

            # Track exercise dates - only update if exercising earlier
            self._exercise_dates[exercise_now] = date

        # Final price: average over evaluation paths
        payoff_0 = payoffs[0, 0]
        price = max(payoff_0, np.mean(values[self.split:]))

        return price, time_path_gen

    def price_upper_lower_bound(self, train_eval_split=2):
        """
        Compute both lower and upper bounds using NLSM with neural networks.

        Lower bound: Regular NLSM pricing
        Upper bound: Dual formulation (Haugh-Kogan method)

        Args:
            train_eval_split: Ratio for splitting paths

        Returns:
            tuple: (lower_bound, upper_bound, time_for_path_generation)
        """
        t_start = time.time()

        # Generate paths
        if configs.path_gen_seed.get_seed() is not None:
            np.random.seed(configs.path_gen_seed.get_seed())
            torch.manual_seed(configs.path_gen_seed.get_seed())

        path_result = self.model.generate_paths()
        if isinstance(path_result, tuple):
            stock_paths, var_paths = path_result
        else:
            stock_paths = path_result
            var_paths = None

        time_path_gen = time.time() - t_start
        print(f"time path gen: {time_path_gen:.4f} ", end="")

        # Compute payoffs
        payoffs = self.payoff(stock_paths)

        # Discount all payoffs upfront (reference methodology)
        power = np.arange(0, self.model.nb_dates + 1)
        disc_factors = np.exp(
            (-self.model.rate) * self.model.maturity / self.model.nb_dates * power
        )
        disc_factors = np.repeat(
            np.expand_dims(disc_factors, axis=0), repeats=payoffs.shape[0], axis=0
        )
        payoffs = payoffs * disc_factors

        # Split
        self.split = len(stock_paths) // train_eval_split

        nb_paths, nb_stocks, nb_dates_from_shape = stock_paths.shape

        # Initialize M_diff for upper bound tracking
        M_diff = np.zeros((nb_paths, self.model.nb_dates + 1))

        # Clear previous learned networks
        self._learned_networks = {}

        # Initialize exercise dates tracking (for get_exercise_time)
        self._exercise_dates = np.full(nb_paths, self.model.nb_dates, dtype=int)

        # Initialize with terminal payoff (already discounted)
        values = payoffs[:, -1].copy()
        prev_cont_val = np.zeros_like(values)

        # Backward induction from T-1 down to 0
        for date in range(self.model.nb_dates - 1, -1, -1):
            immediate_exercise = payoffs[:, date]
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Learn continuation value using neural network
            continuation_values = self._learn_continuation(
                current_state, values, immediate_exercise, date
            )

            # Update M_diff for upper bound (reference methodology)
            if date < self.model.nb_dates - 1:
                M_diff[:, date + 1] = np.maximum(
                    payoffs[:, date + 1], prev_cont_val
                ) - continuation_values

            # Lower bound: update values
            if date > 0:
                exercise_now = immediate_exercise > continuation_values
                values[exercise_now] = immediate_exercise[exercise_now]
                # Track exercise dates
                self._exercise_dates[exercise_now] = date

            prev_cont_val = continuation_values.copy()

        # Compute lower bound
        payoff_0 = payoffs[0, 0]
        lower_bound = max(payoff_0, np.mean(values[self.split:]))

        # Compute upper bound using dual formulation (reference methodology)
        M = np.cumsum(M_diff, axis=1)
        upper_bound = np.mean(np.max(payoffs[self.split:] - M[self.split:], axis=1))

        return lower_bound, upper_bound, time_path_gen

    def _learn_continuation(self, current_state, future_values, immediate_exercise, date):
        """
        Learn continuation value using neural network regression.

        Args:
            current_state: (nb_paths, state_size) - Current stock prices (+ payoff if used)
            future_values: (nb_paths,) - Discounted future values
            immediate_exercise: (nb_paths,) - Immediate payoff if exercised
            date: int - Current time step

        Returns:
            continuation_values: (nb_paths,) - Estimated continuation values
        """
        # Determine which paths to use for training and evaluation (reference methodology)
        if self.train_ITM_only:
            # Only train on in-the-money paths in training set
            train_itm = np.where(immediate_exercise[:self.split] > 0)[0]
            # Only predict on in-the-money paths in ALL paths
            eval_itm = np.where(immediate_exercise > 0)[0]
        else:
            # Use all paths
            train_itm = np.arange(self.split)
            eval_itm = np.arange(len(immediate_exercise))

        # Initialize continuation values to zero (OTM paths stay at 0)
        continuation_values = np.zeros(current_state.shape[0])

        # Reinitialize network weights for this time step (reference methodology)
        self.neural_network.apply(init_weights)

        # Train the network on ITM paths in training set
        if len(train_itm) > 0:
            self._train_network(
                current_state[train_itm],
                future_values[train_itm]
            )

        # Note: Don't store networks for performance (2-5x speedup)

        # Predict only on ITM paths (OTM paths remain 0)
        if len(eval_itm) > 0:
            continuation_values[eval_itm] = self._evaluate_network(current_state[eval_itm])

        return continuation_values

    def _train_network(self, X_inputs, Y_labels, batch_size=2000):
        """
        Train neural network using gradient descent.

        Args:
            X_inputs: Training inputs (stock prices)
            Y_labels: Training targets (continuation values)
            batch_size: Batch size for training
        """
        optimizer = optim.Adam(self.neural_network.parameters())
        X_inputs = torch.from_numpy(X_inputs).double()
        Y_labels = torch.from_numpy(Y_labels).double().view(len(Y_labels), 1)

        self.neural_network.train(True)

        for epoch in range(self.nb_epochs):
            for batch in tdata.BatchSampler(
                tdata.RandomSampler(range(len(X_inputs)), replacement=False),
                batch_size=batch_size,
                drop_last=False
            ):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.neural_network(X_inputs[batch])
                    loss = nn.MSELoss(reduction="mean")(outputs, Y_labels[batch])
                    loss.backward()
                    optimizer.step()

    def _evaluate_network(self, X_inputs):
        """
        Evaluate neural network on inputs.

        Args:
            X_inputs: Input states

        Returns:
            Predicted continuation values
        """
        self.neural_network.train(False)
        X_inputs = torch.from_numpy(X_inputs).double()
        outputs = self.neural_network(X_inputs)
        continuation_values = outputs.view(len(X_inputs)).detach().numpy()

        # Clip to non-negative (American option value can't be negative)
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values

    def get_exercise_time(self):
        """Return average exercise time normalized to [0, 1] (evaluation set only)."""
        if not hasattr(self, '_exercise_dates'):
            return None

        nb_dates = self.model.nb_dates
        # Only use evaluation set paths (self.split:), not training paths
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """
        Apply learned policy using backward induction (same as training).

        NOTE: This method is not supported with the current speed optimization approach.
        Network weights are reinitialized at each timestep and not stored.
        To use this method, networks would need to be stored (2-5x slower).

        Args:
            stock_paths: (nb_paths, nb_stocks, nb_dates+1) - Stock price paths
            var_paths: (nb_paths, nb_stocks, nb_dates+1) - Variance paths (optional)

        Returns:
            exercise_times: (nb_paths,) - Time step when each path is exercised
            payoff_values: (nb_paths,) - Payoff value at exercise for each path
            price: float - Average discounted payoff

        Raises:
            NotImplementedError: Method not supported with current optimization approach
        """
        raise NotImplementedError(
            "backward_induction_on_paths() is not supported with the current speed optimization. "
            "Networks are not stored during training for 2-5x speedup. "
            "To use this method, network storage would need to be re-enabled."
        )

        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        # Compute all payoffs upfront
        payoffs = self.payoff(stock_paths)

        # Initialize with terminal payoff
        values = payoffs[:, -1].copy()

        # Track exercise dates (initialize to maturity = nb_dates)
        exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # Backward induction from T-1 to 1 (same as in price())
        for date in range(nb_dates - 1, 0, -1):
            # Skip if no network learned for this time step
            if date not in self._learned_networks:
                continue

            # Current immediate exercise value
            immediate_exercise = payoffs[:, date]

            # Prepare state (match NLSM's training structure)
            current_state = stock_paths[:, :, date]

            if self.use_payoff_as_input:
                current_state = np.concatenate([current_state, payoffs[:, date:date+1]], axis=1)

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Get learned network for this time step
            self.neural_network = self._learned_networks[date]

            # Compute continuation values using trained network
            continuation_values = self._evaluate_network(current_state)

            # Clip to non-negative (American option value can't be negative)
            continuation_values = np.maximum(0, continuation_values)

            # Discount future values
            discounted_values = values * disc_factor

            # Exercise decision: exercise if immediate > continuation
            exercise_now = immediate_exercise > continuation_values

            # Update values
            values = np.where(exercise_now, immediate_exercise, discounted_values)

            # Track exercise dates - only update if exercising earlier
            exercise_dates[exercise_now] = date

        # Extract payoff values at exercise time
        payoff_values = np.array([payoffs[i, exercise_dates[i]] for i in range(nb_paths)])

        # Compute price (average discounted payoff)
        price = np.mean(values)

        return exercise_dates, payoff_values, price
