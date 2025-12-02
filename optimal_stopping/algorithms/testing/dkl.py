import torch
import gpytorch
import numpy as np
import time
import math
from optimal_stopping.run import configs


# ---------------------------------------------------------
# 1. Define the DKL Model Architecture
# ---------------------------------------------------------

class LargeFeatureExtractor(torch.nn.Sequential):
    """Deep Neural Network Feature Extractor."""

    def __init__(self, data_dim, feature_dim=2, hidden_dim=50):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, hidden_dim))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(hidden_dim, hidden_dim))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(hidden_dim, feature_dim))


class GPRegressionModel(gpytorch.models.ExactGP):
    """Gaussian Process with Deep Kernel."""

    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor[-1].out_features)
        )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------------------------------------------------------
# 2. The Pricer Class
# ---------------------------------------------------------

class DKL_LSM:
    """
    Gaussian Process with Deep Kernel Learning (DKL) for Option Pricing.
    Handles both Standard and Path-Dependent options.
    """

    def __init__(self, model, payoff, nb_epochs=50, hidden_size=50,
                 feature_dim=2, train_ITM_only=True, use_payoff_as_input=False,
                 train_sample_size=1000, **kwargs):
        self.model = model
        self.payoff = payoff
        self.nb_epochs = nb_epochs
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.train_ITM_only = train_ITM_only
        self.use_payoff_as_input = use_payoff_as_input
        self.train_sample_size = train_sample_size

        self.use_var = getattr(model, 'return_var', False)
        # Check if payoff needs history
        self.is_path_dependent = getattr(payoff, 'is_path_dependent', False)

    def price(self, train_eval_split=2):
        t_start = time.time()

        # 1. Generate Paths
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

        self.split = len(stock_paths) // train_eval_split
        nb_paths = stock_paths.shape[0]
        nb_dates = self.model.nb_dates
        disc_factor = math.exp(-self.model.rate * self.model.maturity / nb_dates)

        # 2. Initialize Terminal Values
        if self.is_path_dependent:
            # FIX: Pass full history for path-dependent payoffs
            values = self.payoff.eval(stock_paths)
        else:
            values = self.payoff.eval(stock_paths[:, :, -1])

        # FIX: Robust shape handling
        if values.ndim > 1: values = values[:, -1] if values.shape[1] > 1 else values.reshape(-1)
        values = values.reshape(-1)

        # 3. Backward Induction
        for date in range(nb_dates - 1, 0, -1):

            # --- Calculate Immediate Exercise (Target) ---
            if self.is_path_dependent:
                # FIX: Pass history up to current date
                path_slice = stock_paths[:, :, :date + 1]
                immediate_exercise = self.payoff.eval(path_slice)
            else:
                immediate_exercise = self.payoff.eval(stock_paths[:, :, date])

            # FIX: Robust shape handling for broadcasting errors
            if immediate_exercise.ndim > 1:
                immediate_exercise = immediate_exercise[:, -1] if immediate_exercise.shape[
                                                                      1] > 1 else immediate_exercise.reshape(-1)
            immediate_exercise = immediate_exercise.reshape(-1)

            # --- Prepare State (Input X) ---
            current_state = stock_paths[:, :, date]

            if self.use_var and var_paths is not None:
                current_state = np.concatenate([current_state, var_paths[:, :, date]], axis=1)

            # FIX: Add payoff as input (Crucial for Barriers!)
            if self.use_payoff_as_input:
                current_state = np.concatenate(
                    [current_state, immediate_exercise.reshape(-1, 1)], axis=1
                )

            # --- DKL Step (Train & Predict) ---
            continuation_values = self._learn_dkl(
                current_state,
                values * disc_factor,
                immediate_exercise
            )

            # --- Update Values ---
            exercise_now = immediate_exercise > continuation_values
            values[exercise_now] = immediate_exercise[exercise_now]
            values[~exercise_now] *= disc_factor

        # Final Price at t=0
        if self.is_path_dependent:
            payoff_0 = self.payoff.eval(stock_paths[:, :, :1])
        else:
            payoff_0 = self.payoff.eval(stock_paths[:, :, 0])

        if payoff_0.ndim > 1: payoff_0 = payoff_0[:, -1] if payoff_0.shape[1] > 1 else payoff_0.reshape(-1)
        payoff_0 = payoff_0.reshape(-1)

        final_price = max(payoff_0[0], np.mean(values[self.split:]) * disc_factor)

        return final_price, time_path_gen

    def _learn_dkl(self, X, Y, immediate_exercise):
        """
        Train Deep Kernel GP and predict continuation values.
        """
        if self.train_ITM_only:
            itm_mask = (immediate_exercise > 0)
        else:
            itm_mask = np.ones(len(immediate_exercise), dtype=bool)

        if itm_mask.sum() == 0:
            return np.zeros(len(Y))

        X_itm = X[itm_mask]
        Y_itm = Y[itm_mask]

        # Subsampling for Training (O(N^3) fix)
        n_samples = X_itm.shape[0]
        if n_samples > self.train_sample_size:
            indices = np.random.choice(n_samples, self.train_sample_size, replace=False)
            train_x = torch.from_numpy(X_itm[indices]).float()
            train_y = torch.from_numpy(Y_itm[indices]).float()
        else:
            train_x = torch.from_numpy(X_itm).float()
            train_y = torch.from_numpy(Y_itm).float()

        # Initialize Model
        input_dim = train_x.shape[1]
        feature_extractor = LargeFeatureExtractor(
            input_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_size
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor)

        # Train
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.nb_epochs):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Prediction
        model.eval()
        likelihood.eval()

        full_x_itm = torch.from_numpy(X_itm).float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(full_x_itm))
            pred_mean = observed_pred.mean.numpy()

        continuation_values = np.zeros(len(Y))
        continuation_values[itm_mask] = pred_mean

        return np.maximum(0, continuation_values)