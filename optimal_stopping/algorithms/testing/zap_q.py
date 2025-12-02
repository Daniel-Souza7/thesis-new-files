import numpy as np
import time
import math
import torch
from optimal_stopping.run import configs


class ZapQ:
    """
    Zap Q-Learning for Optimal Stopping.

    Implements Algorithm 1 from "Zap Q-Learning for Optimal Stopping Time Problems"
    (Chen, Devraj, Busic, Meyn, 2021).

    Features:
    - Standalone (No inheritance from LSM/RLSM).
    - Universal: Handles both Standard and Path-Dependent options via internal checks.
    - No Neural Networks: Uses Polynomial basis functions for feature mapping.
    - Stochastic Approximation: Uses the Matrix Gain (A_hat) update rule.
    """

    def __init__(self, model, payoff, nb_epochs=None, hidden_size=None,
                 factors=None, train_ITM_only=True, use_payoff_as_input=False,
                 poly_degree=3, zap_p=0.85, zap_q=0.85, zap_gain=10.0, **kwargs):
        """
        Initialize ZapQ Pricer.

        Args:
            model: Stock model
            payoff: Payoff object
            train_ITM_only: If True, only updates on In-The-Money paths
            use_payoff_as_input: If True, appends payoff value to state
            poly_degree: Degree of polynomial basis functions (default: 3)
            zap_p: Learning rate decay power (0.5 < p <= 1)
            zap_q: Matrix gain decay power (0.5 < q <= 1)
            zap_gain: Initial gain scaling factor
        """
        self.model = model
        self.payoff = payoff
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.use_var = getattr(model, 'return_var', False)

        # Basis Function Settings
        self.poly_degree = poly_degree

        # Zap Hyperparameters
        self.zap_p = zap_p
        self.zap_q = zap_q
        self.g_step = zap_gain

        # Check path dependency once at init
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

    def get_features(self, state):
        """
        Map raw state to Polynomial Features (No Neural Networks).

        Args:
            state: (N, D) array of raw inputs (Stock prices, etc.)

        Returns:
            psi: (N, K) array of features.
                 Includes [1, S, S^2, S^3, ...] for each dimension.
        """
        # 1. Constant term (Bias)
        N = state.shape[0]
        features = [np.ones((N, 1))]

        # 2. Polynomial terms for each dimension
        # Normalize state slightly to prevent explosion with high degrees
        norm_state = state / self.model.spot

        for d in range(1, self.poly_degree + 1):
            features.append(norm_state ** d)

        # 3. Concatenate
        psi = np.concatenate(features, axis=1)
        return psi

    def price(self, train_eval_split=2):
        """
        Run the pricing algorithm using Backward Induction + Zap Q-Learning.
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
        nb_dates = self.model.nb_dates  # Actual steps

        # Calculate discount factor per step
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # 2. Initialize Terminal Values
        # Handle Path Dependence for Terminal Payoff
        if self.is_path_dependent:
            values = self.payoff.eval(stock_paths)
        else:
            values = self.payoff.eval(stock_paths[:, :, -1])

        # Flatten shapes if necessary
        if values.ndim > 1: values = values.reshape(-1)

        # 3. Backward Induction Loop
        for date in range(nb_dates - 1, 0, -1):

            # --- A. Prepare Current State and Immediate Payoff ---
            current_state = stock_paths[:, :, date]

            # Variance
            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # Calculate Immediate Exercise (Target for Decision)
            if self.is_path_dependent:
                path_history = stock_paths[:, :, :date + 1]
                immediate_exercise = self.payoff.eval(path_history)
            else:
                immediate_exercise = self.payoff.eval(stock_paths[:, :, date])

            if immediate_exercise.ndim > 1: immediate_exercise = immediate_exercise.reshape(-1)

            # Payoff as Input Feature (Optional)
            if self.use_payoff_as_input:
                current_state = np.concatenate(
                    [current_state, immediate_exercise.reshape(-1, 1)], axis=1
                )

            # --- B. Run Zap Q-Learning Step ---
            # This replaces the standard Least Squares regression
            continuation_values = self._zap_update_step(
                current_state,
                values * disc_factor,  # Target: Discounted Future Value
                immediate_exercise
            )

            # --- C. Update Values (Optimal Stopping Decision) ---
            exercise_now = immediate_exercise > continuation_values

            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

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

        Instead of batch OLS (inv(X'X)X'Y), we update theta iteratively.
        """
        # 1. Generate Basis Features (Psi)
        psi = self.get_features(current_state)

        # 2. Filter for ITM paths (Training Set)
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        psi_train = psi[:self.split][train_mask]
        target_train = future_values[:self.split][train_mask]

        n_samples = psi_train.shape[0]
        d = psi_train.shape[1]  # Dimension of features

        # If no ITM paths, return zeros
        if n_samples == 0:
            return np.zeros(current_state.shape[0])

        # 3. Initialization for Zap
        theta = np.zeros(d)
        # Initialize A_hat as negative identity (for stability)
        A_hat = -1.0 * np.eye(d)

        # 4. Stochastic Approximation Loop
        # Iterate through the batch as if it were a stream
        for n in range(n_samples):
            step_idx = n + 1

            # a. Step sizes (decreasing)
            alpha_n = self.g_step / (step_idx ** self.zap_p)
            gamma_n = self.g_step / (step_idx ** self.zap_q)

            # b. Get sample
            psi_n = psi_train[n]
            y_n = target_train[n]

            # c. Prediction and Error (TD Error)
            # d_n+1 in paper
            prediction = np.dot(theta, psi_n)
            error = y_n - prediction

            # d. Update Matrix Gain Estimate (A_hat)
            # A_n+1 = -psi * psi.T (Negative Definite)
            # Update: A_hat_new = A_hat + gamma * (A_next - A_hat)
            outer_prod = -1.0 * np.outer(psi_n, psi_n)
            A_hat = A_hat + gamma_n * (outer_prod - A_hat)

            # e. Update Parameters (Theta)
            # theta_new = theta - alpha * A_hat_inv * psi * error
            # Solve linear system A_hat * update = psi * error
            rhs = psi_n * error

            try:
                # Using solve is numerically more stable than inv()
                update_dir = np.linalg.solve(A_hat, rhs)
            except np.linalg.LinAlgError:
                # Fallback regularization if singular
                update_dir = np.linalg.solve(A_hat - 1e-6 * np.eye(d), rhs)

            # Note: Formula is minus, but A_hat is negative definite,
            # so this moves in the correct gradient direction.
            theta = theta - alpha_n * update_dir

        # 5. Predict for ALL paths (Evaluation)
        continuation_values = np.dot(psi, theta)

        # American Option constraint
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values