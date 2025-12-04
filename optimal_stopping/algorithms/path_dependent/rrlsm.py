import torch
import numpy as np
import time
import math
from optimal_stopping.run import configs
from optimal_stopping.algorithms.utils import randomized_neural_networks


class RRLSM:
    """
    Recurrent Randomized Least Squares Monte Carlo (RRLSM).
    Standalone class (No inheritance).

    Uses randomRNN (Echo State Networks) to capture path-dependent dynamics.
    """

    def __init__(self, model, payoff, hidden_size=100, factors=(1., 1., 1.),
                 train_ITM_only=True, use_payoff_as_input=False, **kwargs):
        """
        Initialize RRLSM pricer.
        """
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.factors = factors
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input

        # Check for variance paths (needed for Rough Heston)
        self.use_var = getattr(model, 'return_var', False)

        # Initialize hidden size default
        if hidden_size < 0:
            hidden_size = 50 + abs(hidden_size) * model.nb_stocks

        # Calculate input state size
        # nb_stocks + (1 if variance used) + (1 if payoff input used)
        state_size = model.nb_stocks * (1 + self.use_var) + self.use_payoff_as_input * 1

        # -------------------------------------------------------
        # Initialize randomRNN
        # -------------------------------------------------------
        # randomRNN expects 3 factors: (Input_scale, Hidden_scale, Aux_scale)
        rnn_factors = list(factors)
        while len(rnn_factors) < 3:
            rnn_factors.append(1.0)  # Default scale for missing factors

        self.reservoir = randomized_neural_networks.randomRNN(
            hidden_size,
            state_size,
            factors=tuple(rnn_factors),
            extend=False
        )

        # The basis functions are the output of the RNN (+1 for constant)
        self.nb_base_fcts = self.reservoir.hidden_size + 1

        # Storage for learned policy
        self._learned_coefficients = {}

    def price(self, train_eval_split=2):
        """
        Compute price using RNN-based features with robust shape handling.
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

        self.split = len(stock_paths) // train_eval_split
        nb_paths, nb_stocks, nb_dates_plus_one = stock_paths.shape
        nb_dates = nb_dates_plus_one - 1
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # ---------------------------------------------------------
        # NEW STEP: Pre-calculate RNN Memory States (Forward Pass)
        # ---------------------------------------------------------

        # 1. Prepare Input Tensor: Transpose to (Time, Batch, Stocks)
        X_input = stock_paths[:, :self.model.nb_stocks, :].transpose(2, 0, 1)

        # 2. Add variance if available
        if self.use_var and var_paths is not None:
            V_input = var_paths.transpose(2, 0, 1)
            X_input = np.concatenate([X_input, V_input], axis=2)

        # 3. Add Payoff as input if requested
        if self.use_payoff_as_input:
            payoff_vals = np.zeros((nb_paths, 1, nb_dates_plus_one))

            for t in range(nb_dates_plus_one):
                path_slice = stock_paths[:, :, :t + 1]
                val = self.payoff.eval(path_slice)

                # FIX 1: Robust Shape Handling
                if val.ndim > 1 and val.shape[1] > 1:
                    val = val[:, -1]

                # Enforce flat shape (N,)
                payoff_vals[:, 0, t] = val.reshape(nb_paths)

            # Transpose to (Time, Batch, 1) and concatenate
            payoff_input = payoff_vals.transpose(2, 0, 1)
            X_input = np.concatenate([X_input, payoff_input], axis=2)

        # 4. Convert to Tensor
        X_tensor = torch.from_numpy(X_input).type(torch.float32)

        # 5. Run the RNN
        # print("Computing RNN States...", end="")
        all_hidden_states = self.reservoir(X_tensor).detach().numpy()
        # print("Done.")
        # ---------------------------------------------------------

        # Initialize with terminal payoff
        values = self.payoff.eval(stock_paths)
        # FIX 2: Handle shape for terminal values
        if values.ndim > 1 and values.shape[1] > 1:
            values = values[:, -1]
        values = values.reshape(nb_paths)

        self._exercise_dates = np.full(nb_paths, nb_dates, dtype=int)
        self._learned_coefficients = {}

        # Backward Induction
        for date in range(nb_dates - 1, 0, -1):
            # Payoff Evaluation
            path_history = stock_paths[:, :, :date + 1]
            immediate_exercise = self.payoff.eval(path_history)

            # FIX 3: Handle shape inside loop
            if immediate_exercise.ndim > 1 and immediate_exercise.shape[1] > 1:
                immediate_exercise = immediate_exercise[:, -1]
            immediate_exercise = immediate_exercise.reshape(nb_paths)

            # Regression on RNN_Hidden_State
            current_rnn_features = all_hidden_states[date, :, :]

            # Learn continuation
            continuation_values, coefficients = self._learn_continuation_rnn(
                current_rnn_features,
                values * disc_factor,
                immediate_exercise
            )

            self._learned_coefficients[date] = coefficients

            # Update Values
            exercise_now = immediate_exercise > continuation_values

            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor
            self._exercise_dates[exercise_now] = date

        # Final payoff t=0
        payoff_0 = self.payoff.eval(stock_paths[:, :, :1])
        # FIX 4: Handle shape for t=0
        if payoff_0.ndim > 1 and payoff_0.shape[1] > 1:
            payoff_0 = payoff_0[:, -1]
        payoff_0 = payoff_0.reshape(nb_paths)

        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

    def _learn_continuation_rnn(self, features, future_values, immediate_exercise):
        """
        Simplified regression using pre-calculated RNN features.
        """
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        basis = np.concatenate([features, np.ones((len(features), 1))], axis=1)

        if train_mask.sum() > 0:
            coefficients = np.linalg.lstsq(
                basis[:self.split][train_mask],
                future_values[:self.split][train_mask],
                rcond=None
            )[0]
        else:
            coefficients = np.zeros(basis.shape[1])

        continuation_values = np.dot(basis, coefficients)
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values, coefficients

    def get_exercise_time(self):
        """Return average exercise time."""
        if not hasattr(self, '_exercise_dates'):
            return None
        nb_dates = self.model.nb_dates
        normalized_times = self._exercise_dates[self.split:] / nb_dates
        return float(np.mean(normalized_times))