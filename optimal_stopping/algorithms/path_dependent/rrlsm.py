class RecurrentSRLSM(SRLSM):
    """
    Recurrent RLSM: Uses randomRNN (Echo State Network) instead of Reservoir2.

    Best for:
    - Rough Heston / Rough Volatility (Non-Markovian dynamics)
    - Path dependent options where the 'state' is complex
    """

    def _init_reservoir(self, state_size, hidden_size, factors):
        """Override to use randomRNN."""
        # We need the 3 factors for RNN (Input, Hidden, Aux)
        # If user only provided 2, we default the 3rd to 1.0
        rnn_factors = factors[1:]
        if len(rnn_factors) < 3:
            rnn_factors = (factors[1], factors[2], 1.0)  # Input_scale, Hidden_scale, Aux_scale

        self.reservoir = randomized_neural_networks.randomRNN(
            hidden_size,
            state_size,
            factors=rnn_factors,
            extend=False  # Set True if you want the auxiliary path
        )
        self.nb_base_fcts = self.reservoir.hidden_size  # RNN output size

    def price(self, train_eval_split=2):
        """
        Modified price function to pre-calculate RNN states.
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
        # We need to feed the whole sequence (0 to T) to the RNN
        # stock_paths shape: (Batch, Stocks, Time) -> Need (Time, Batch, Stocks)

        # Prepare input tensor
        X_input = stock_paths[:, :self.model.nb_stocks, :].transpose(2, 0, 1)  # (Time, Batch, Stocks)

        if self.use_var and var_paths is not None:
            V_input = var_paths.transpose(2, 0, 1)
            X_input = np.concatenate([X_input, V_input], axis=2)

        X_tensor = torch.from_numpy(X_input).type(torch.float32)

        # Get all hidden states at once: (Time, Batch, Hidden_Size)
        # This is where the "Interconnection" (Memory) happens!
        self.all_hidden_states = self.reservoir(X_tensor).detach().numpy()
        # ---------------------------------------------------------

        # Initialize with terminal payoff
        values = self.payoff.eval(stock_paths)
        self._exercise_dates = np.full(nb_paths, nb_dates, dtype=int)
        self._learned_coefficients = {}

        # Backward Induction
        for date in range(nb_dates - 1, 0, -1):
            path_history = stock_paths[:, :, :date + 1]
            immediate_exercise = self.payoff.eval(path_history)

            # NEW: Instead of current_state, we get the pre-calculated RNN state
            # Shape: (Batch, Hidden_Size)
            current_rnn_features = self.all_hidden_states[date, :, :]

            # Learn continuation (modified to accept features directly)
            continuation_values, coefficients = self._learn_continuation_rnn(
                current_rnn_features,
                values * disc_factor,
                immediate_exercise
            )

            self._learned_coefficients[date] = coefficients

            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor
            self._exercise_dates[exercise_now] = date

        # Final payoff t=0
        payoff_0 = self.payoff.eval(stock_paths[:, :, :1])
        return max(payoff_0[0], np.mean(values[self.split:]) * disc_factor), time_path_gen

    def _learn_continuation_rnn(self, features, future_values, immediate_exercise):
        """
        Modified regression: Features are already computed by RNN.
        """
        if self.train_ITM_only:
            train_mask = (immediate_exercise[:self.split] > 0)
        else:
            train_mask = np.ones(self.split, dtype=bool)

        # Features are already passed in, no need to call self.reservoir()
        # Just add constant term
        basis = np.concatenate([features, np.ones((len(features), 1))], axis=1)

        # Regression
        coefficients = np.linalg.lstsq(
            basis[:self.split][train_mask],
            future_values[:self.split][train_mask],
            rcond=None
        )[0]

        continuation_values = np.dot(basis, coefficients)
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values, coefficients