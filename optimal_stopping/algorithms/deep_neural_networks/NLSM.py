import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"


class NeuralLSM:
    def __init__(self, model, payoff, hidden_size=20, nb_paths=8192,
                 nb_epochs=10, learning_rate=0.001, use_payoff_as_input=True,
                 train_ITM_only=True, batch_size=512, **kwargs):
        self.model = model
        self.payoff = payoff
        self.hidden_size = hidden_size
        self.nb_paths = nb_paths

        # Optimization: Use full epochs for T-1, 1/5th for others (Weight Transfer)
        self.nb_epochs = nb_epochs
        self.nb_epochs_update = max(1, nb_epochs // 5)

        self.learning_rate = learning_rate
        self.use_payoff_as_input = use_payoff_as_input
        self.train_ITM_only = train_ITM_only
        self.batch_size = batch_size

        self.r = model.rate
        self.dt = model.dt
        self.nb_dates = model.nb_dates
        self.nb_stocks = model.nb_stocks
        self.S0 = np.array(model.spot).flatten()

        if len(self.S0) == 1 and self.nb_stocks > 1:
            self.S0 = np.full(self.nb_stocks, self.S0[0])

        self.input_dim = self.nb_stocks + (1 if self.use_payoff_as_input else 0)
        self.networks = {}
        self.feature_mean = None
        self.feature_std = None

    def _generate_paths_safe(self, nb_paths):
        """Fixes the 'tuple' error by extracting paths from the model output."""
        start_time = time.time()
        result = self.model.generate_paths(nb_paths=nb_paths)
        elapsed = time.time() - start_time

        # If model returns (paths, time), take [0]. If just paths, take result.
        paths = result[0] if isinstance(result, tuple) else result
        return paths, elapsed

    def _build_features(self, states, payoffs=None, fit=False):
        # 1. Calculate log-returns for stocks (e.g., shape [8192, 2])
        # Ensure S0 is reshaped correctly for broadcasting
        features = np.log(states / self.S0.reshape(1, -1))

        # 2. Add Payoff column if requested (making shape [8192, 3])
        if self.use_payoff_as_input and payoffs is not None:
            # Normalize payoff by strike to keep it in a similar range to log-returns
            norm_payoffs = (payoffs[:, np.newaxis] / self.payoff.strike).astype(np.float32)
            features = np.concatenate([features, norm_payoffs], axis=1)

        # 3. Fit or Apply Standardization
        if fit:
            # Calculate mean/std across the concatenated features (dim=3)
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8

        if self.feature_mean is not None:
            # This will now work because feature_mean has shape (3,)
            # and features has shape (8192, 3)
            features = (features - self.feature_mean) / self.feature_std

        return features.astype(np.float32)

    def _get_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        ).to(DEVICE)

    def _train_step(self, net, loader, epochs):
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = net(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        return net

    def _train_continuation_networks(self, paths, payoffs_all):
        n_paths = paths.shape[0]
        T = self.nb_dates
        df = self.model.df
        V = np.zeros((n_paths, T + 1), dtype=np.float32)
        V[:, T] = payoffs_all[:, T]

        # Fit normalizer once
        # Pass payoffs during fit so the scaler knows about the 3rd column
        self._build_features(paths[:, :, T // 2], payoffs=payoffs_all[:, T // 2], fit=True)
        current_net = self._get_network()

        for t in range(T - 1, -1, -1):
            states = paths[:, :, t]
            payoffs_t = payoffs_all[:, t]
            features = self._build_features(states, payoffs_t)
            targets = (df * V[:, t + 1]).astype(np.float32)

            if self.train_ITM_only:
                itm_mask = payoffs_t > 0
                train_x, train_y = (features[itm_mask], targets[itm_mask]) if np.sum(itm_mask) > self.batch_size else (
                    features, targets)
            else:
                train_x, train_y = features, targets

            loader = DataLoader(TensorDataset(torch.from_numpy(train_x).to(DEVICE),
                                              torch.from_numpy(train_y).to(DEVICE)),
                                batch_size=self.batch_size, shuffle=True)

            # SPEED BOOSTER: Weight Transfer
            epochs = self.nb_epochs if t == T - 1 else self.nb_epochs_update
            current_net = self._train_step(current_net, loader, epochs)

            self.networks[t] = {k: v.cpu().clone() for k, v in current_net.state_dict().items()}

            current_net.eval()
            with torch.no_grad():
                cont_values = current_net(torch.from_numpy(features).to(DEVICE)).cpu().numpy().flatten()

            cont_values = np.maximum(0, cont_values)
            stop_now = (payoffs_t >= cont_values) & (payoffs_t > 0)
            V[:, t] = np.where(stop_now, payoffs_t, cont_values)

    def _price_with_networks(self, paths, payoffs_all):
        n_paths, T = paths.shape[0], self.nb_dates
        df = self.model.df
        stopped, path_values = np.zeros(n_paths, dtype=bool), np.zeros(n_paths, dtype=np.float32)
        eval_net = self._get_network()
        eval_net.eval()

        for t in range(T):
            if np.all(stopped): break
            eval_net.load_state_dict(self.networks[t])
            features = self._build_features(paths[:, :, t], payoffs_all[:, t])
            with torch.no_grad():
                cont_values = eval_net(torch.from_numpy(features).to(DEVICE)).cpu().numpy().flatten()

            exercise_now = (payoffs_all[:, t] >= cont_values) & (payoffs_all[:, t] > 0) & (~stopped)
            path_values[exercise_now] = payoffs_all[exercise_now, t] * (df ** t)
            stopped[exercise_now] = True

        path_values[~stopped] = payoffs_all[~stopped, T] * (df ** T)
        return np.mean(path_values)

    def price(self, **kwargs):
        # 1. Train paths
        train_paths, t1 = self._generate_paths_safe(self.nb_paths)
        train_payoffs = self.payoff(train_paths)
        self._train_continuation_networks(train_paths, train_payoffs)

        # 2. Eval paths
        eval_paths, t2 = self._generate_paths_safe(self.nb_paths)
        eval_payoffs = self.payoff(eval_paths)
        price = self._price_with_networks(eval_paths, eval_payoffs)

        return float(price), float(t1 + t2)