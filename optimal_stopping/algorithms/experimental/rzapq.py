import numpy as np
import time
import math
import torch
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class RZapQ:
    """
    Zap Q-Learning for Optimal Stopping using Randomized Neural Networks.

    Features:
    - Standalone (No inheritance from LSM/RLSM).
    - Universal: Handles both Standard and Path-Dependent options.
    - Randomized NN Feature Map: Uses a fixed, randomly initialized hidden layer.
    - Stochastic Approximation: Uses the Zap Matrix Gain (A_hat) update rule.
    """

    def __init__(self, model, payoff, hidden_size=100, factors=(1., 1.),
                 train_ITM_only=True, use_payoff_as_input=False,
                 zap_p=0.85, zap_q=0.85, zap_gain=10.0, **kwargs):
        """
        Initialize ZapQRNN Pricer.
        """
        self.model = model
        self.payoff = payoff
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)

        # RNN Hyperparameters
        self.hidden_size = hidden_size
        self.factors = factors

        # Zap Hyperparameters
        self.zap_p = zap_p
        self.zap_q = zap_q
        self.g_step = zap_gain

        # Check path dependency
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

        # --- Initialize Reservoir ---
        # Calculate input dimension size based on config
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

        self.reservoir = randomized_neural_networks.Reservoir2(
            hidden_size,
            state_size,
            factors=factors[1:],  # Scaling factors
            activation=torch.nn.LeakyReLU(factors[0] / 2)  # Activation function
        )

    def get_features(self, state):
        """
        Map raw state to features using Randomized Neural Network.
        """
        X_tensor = torch.from_numpy(state).type(torch.float32)
        psi = self.reservoir(X_tensor).detach().numpy()
        psi = np.concatenate([psi, np.ones((len(psi), 1))], axis=1)
        return psi

    def price(self, train_eval_split=2):
        """
        Run the pricing algorithm using Backward Induction + Zap Q-Learning (RNN).
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

        # Setup split
        self.split = len(stock_paths) // train_eval_split
        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates

        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # 2. Initialize Terminal Values
        if self.is_path_dependent:
            values = self.payoff.eval(stock_paths)
        else:
            values = self.payoff.eval(stock_paths[:, :, -1])

        if values.ndim > 1: values = values.reshape(-1)

        # NEW: Initialize exercise tracking (Default = Maturity)
        self._exercise_dates = np.full(nb_paths, nb_dates, dtype=int)

        # 3. Backward Induction Loop
        for date in range(nb_dates - 1, 0, -1):

            # --- A. Prepare Current State and Immediate Payoff ---
            current_state = stock_paths[:, :, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Calculate Immediate Exercise
            if self.is_path_dependent:
                path_history = stock_paths[:, :, :date + 1]
                immediate_exercise = self.payoff.eval(path_history)
            else:
                immediate_exercise = self.payoff.eval(stock_paths[:, :, date])

            if immediate_exercise.ndim > 1: immediate_exercise = immediate_exercise.reshape(-1)

            # Payoff as Input Feature
            if self.use_payoff_as_input:
                current_state = np.concatenate(
                    [current_state, immediate_exercise.reshape(-1, 1)], axis=1
                )

            # --- B. Run Zap Q-Learning Update Step ---
            continuation_values = self._zap_update_step(
                current_state,
                values * disc_factor,  # Target
                immediate_exercise
            )

            # --- C. Update Values (Optimal Stopping Decision) ---
            exercise_now = immediate_exercise > continuation_values

            # Update values and record exercise time
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

            # Record exercise date for paths that exercised NOW
            self._exercise_dates[exercise_now] = date

        # 4. Final Price at t=0
        if self.is_path_dependent:
            payoff_0 = self.payoff.eval(stock_paths[:, :, :1])
        else:
            payoff_0 = self.payoff.eval(stock_paths[:, :, 0])

        if payoff_0.ndim > 1: payoff_0 = payoff_0.reshape(-1)

        # Price on Evaluation Set
        final_price = max(payoff_0[0], np.mean(values[self.split:]) * disc_factor)

        return final_price, time_path_gen

    def _zap_update_step(self, current_state, future_values, immediate_exercise):
        """
        The core Zap Q-Learning update loop (Algorithm 1).
        """
        # 1. Generate Basis Features (Psi) using RNN
        psi = self.get_features(current_state)

        # 2. Filter for ITM paths (Training Set)
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        psi_train = psi[:self.split][train_mask]
        target_train = future_values[:self.split][train_mask]

        n_samples = psi_train.shape[0]
        d = psi_train.shape[1]

        # If no ITM paths, return zeros
        if n_samples == 0:
            return np.zeros(current_state.shape[0])

        # 3. Initialization for Zap
        theta = np.zeros(d)
        A_hat = -1.0 * np.eye(d)

        # 4. Stochastic Approximation Loop
        for n in range(n_samples):
            step_idx = n + 1

            alpha_n = self.g_step / (step_idx ** self.zap_p)
            gamma_n = self.g_step / (step_idx ** self.zap_q)

            psi_n = psi_train[n]
            y_n = target_train[n]

            # Prediction and Error
            prediction = np.dot(theta, psi_n)
            error = y_n - prediction

            # Update Matrix Gain
            outer_prod = -1.0 * np.outer(psi_n, psi_n)
            A_hat = A_hat + gamma_n * (outer_prod - A_hat)

            # Update Parameters
            rhs = psi_n * error
            try:
                update_dir = np.linalg.solve(A_hat, rhs)
            except np.linalg.LinAlgError:
                update_dir = np.linalg.solve(A_hat - 1e-6 * np.eye(d), rhs)

            theta = theta - alpha_n * update_dir

        # 5. Predict for ALL paths
        continuation_values = np.dot(psi, theta)
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