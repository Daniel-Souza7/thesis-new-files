import torch
import gpytorch
import numpy as np
from optimal_stopping.algorithms.experimental.dkl import DKL_LSM, LargeFeatureExtractor, GPRegressionModel


class RandDKL_LSM(DKL_LSM):
    """
    Randomized Deep Kernel Learning.

    Difference: The Neural Network Feature Extractor is RANDOMLY INITIALIZED
    and FROZEN. We only optimize the GP Hyperparameters (lengthscale, noise).
    """

    def _learn_dkl(self, X, Y, immediate_exercise):
        # ... (Data prep same as base class) ...
        if self.train_ITM_only:
            itm_mask = (immediate_exercise > 0)
        else:
            itm_mask = np.ones(len(immediate_exercise), dtype=bool)

        if itm_mask.sum() == 0:
            return np.zeros(len(Y))

        X_itm = X[itm_mask]
        Y_itm = Y[itm_mask]

        n_samples = X_itm.shape[0]
        if n_samples > self.train_sample_size:
            indices = np.random.choice(n_samples, self.train_sample_size, replace=False)
            train_x = torch.from_numpy(X_itm[indices]).float()
            train_y = torch.from_numpy(Y_itm[indices]).float()
        else:
            train_x = torch.from_numpy(X_itm).float()
            train_y = torch.from_numpy(Y_itm).float()

        # 3. Initialize Model
        input_dim = train_x.shape[1]
        feature_extractor = LargeFeatureExtractor(
            input_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_size
        )

        # --- CRITICAL CHANGE: Freeze Neural Network Weights ---
        for param in feature_extractor.parameters():
            param.requires_grad = False
        # ------------------------------------------------------

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor)

        # 4. Train (Optimize ONLY GP Hyperparams)
        model.train()
        likelihood.train()

        # Optimizer only sees GP parameters, not NN parameters
        optimizer = torch.optim.Adam([
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=0.1)  # Higher LR because we are only tuning hyperparams

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.nb_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # 5. Prediction
        model.eval()
        likelihood.eval()

        full_x_itm = torch.from_numpy(X_itm).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(full_x_itm))
            pred_mean = observed_pred.mean.numpy()

        continuation_values = np.zeros(len(Y))
        continuation_values[itm_mask] = pred_mean
        continuation_values = np.maximum(0, continuation_values)

        return continuation_values