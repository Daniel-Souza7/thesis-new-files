""" Underlying model of the stochastic processes that are used:
- Black Scholes
- Heston
- Fractional Brownian motion
- Rough Heston
- Real Data (block bootstrap)
"""

import math
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import scipy.special as scispe

import joblib

NB_JOBS_PATH_GEN = 1


class Model:
    def __init__(self, drift, dividend, volatility, spot, nb_stocks,
                 nb_paths, nb_dates, maturity, name, **keywords):
        self.name = name
        self.drift = drift - dividend
        self.rate = drift  # Store original drift as rate for discounting
        self.dividend = dividend
        self.volatility = volatility
        self.spot = spot
        self.nb_stocks = nb_stocks
        self.nb_paths = nb_paths
        self.nb_dates = nb_dates
        self.maturity = maturity
        self.dt = self.maturity / self.nb_dates
        self.df = math.exp(-self.rate * self.dt)  # Use rate, not drift
        self.return_var = False

    def disc_factor(self, date_begin, date_end):
        """Compute discount factor between two dates."""
        time = (date_end - date_begin) * self.dt
        return math.exp(-self.rate * time)  # FIX: Use self.rate, not self.drift

    def drift_fct(self, x, t):
        """Drift function for SDE."""
        raise NotImplementedError("Subclasses must implement drift_fct()")

    def diffusion_fct(self, x, t, v=0):
        """Diffusion function for SDE."""
        raise NotImplementedError("Subclasses must implement diffusion_fct()")

    def generate_one_path(self):
        """Generate a single path."""
        raise NotImplementedError("Subclasses must implement generate_one_path()")

    def generate_paths(self, nb_paths=None):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates+1) with prices."""
        nb_paths = nb_paths or self.nb_paths
        if NB_JOBS_PATH_GEN > 1:
            return np.array(
                joblib.Parallel(n_jobs=NB_JOBS_PATH_GEN, prefer="threads")(
                    joblib.delayed(self.generate_one_path)()
                    for i in range(nb_paths)))
        else:
            return np.array([self.generate_one_path() for i in range(nb_paths)]), \
                None


class BlackScholes(Model):
    def __init__(self, drift, volatility, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        super(BlackScholes, self).__init__(
            drift=drift, dividend=dividend, volatility=volatility,
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, name="BlackScholes")

    def drift_fct(self, x, t):
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        del t
        return self.volatility * x

    def generate_paths(self, nb_paths=None, return_dW=False, dW=None, X0=None,
                       nb_dates=None):
        """Returns a nparray (nb_paths * nb_stocks * nb_dates+1) with prices."""
        nb_paths = nb_paths or self.nb_paths
        nb_dates = nb_dates or self.nb_dates
        spot_paths = np.empty((nb_paths, self.nb_stocks, nb_dates + 1))

        # Set initial values
        if X0 is None:
            spot_paths[:, :, 0] = self.spot
        else:
            spot_paths[:, :, 0] = X0

        # Generate or use provided Brownian increments
        if dW is None:
            random_numbers = np.random.normal(
                0, 1, (nb_paths, self.nb_stocks, nb_dates))
            dW = random_numbers * np.sqrt(self.dt)

        # Vectorized path generation
        drift = self.drift
        r = np.repeat(np.repeat(np.repeat(
            np.reshape(drift, (-1, 1, 1)), nb_paths, axis=0),
            self.nb_stocks, axis=1), nb_dates, axis=2)
        sig = np.repeat(np.repeat(np.repeat(
            np.reshape(self.volatility, (-1, 1, 1)), nb_paths, axis=0),
            self.nb_stocks, axis=1), nb_dates, axis=2)

        spot_paths[:, :, 1:] = np.repeat(
            spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
            r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))

        # dimensions: [nb_paths, nb_stocks, nb_dates+1]
        if return_dW:
            return spot_paths, None, dW
        return spot_paths, None

    def generate_paths_with_alternatives(
            self, nb_paths=None, nb_alternatives=1, nb_dates=None):
        """
        Generate paths with alternative scenarios for Greeks computation.
        Creates additional paths that branch off at different time points.
        """
        nb_paths = nb_paths or self.nb_paths
        nb_dates = nb_dates or self.nb_dates
        total_nb_paths = nb_paths + nb_paths * nb_alternatives * nb_dates
        spot_paths = np.empty((total_nb_paths, self.nb_stocks, nb_dates + 1))
        spot_paths[:, :, 0] = self.spot
        random_numbers = np.random.normal(
            0, 1, (total_nb_paths, self.nb_stocks, nb_dates))
        mult = nb_alternatives * nb_paths

        # Reuse Brownian increments for branching scenarios
        for i in range(nb_dates - 1):
            random_numbers[
                nb_paths + i * mult:nb_paths + (i + 1) * mult, :, :nb_dates - i - 1] = np.tile(
                random_numbers[:nb_paths, :, :nb_dates - i - 1],
                reps=(nb_alternatives, 1, 1))
        dW = random_numbers * np.sqrt(self.dt)

        drift = self.drift
        r = np.repeat(np.repeat(np.repeat(
            np.reshape(drift, (-1, 1, 1)), total_nb_paths, axis=0),
            self.nb_stocks, axis=1), nb_dates, axis=2)
        sig = np.repeat(np.repeat(np.repeat(
            np.reshape(self.volatility, (-1, 1, 1)), total_nb_paths, axis=0),
            self.nb_stocks, axis=1), nb_dates, axis=2)

        spot_paths[:, :, 1:] = np.repeat(
            spot_paths[:, :, 0:1], nb_dates, axis=2) * np.exp(np.cumsum(
            r * self.dt - (sig ** 2) * self.dt / 2 + sig * dW, axis=2))

        return spot_paths, None


class FractionalBlackScholes(Model):
    def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        super(FractionalBlackScholes, self).__init__(
            drift=drift, dividend=dividend, volatility=volatility,
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, name="FractionalBlackScholes"
        )
        self.hurst = hurst
        self.fBM = FBM(n=nb_dates, hurst=self.hurst, length=maturity, method='cholesky')

    def generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates+1) with prices."""
        path = np.empty((self.nb_stocks, self.nb_dates + 1))
        fracBM_noise = np.empty((self.nb_stocks, self.nb_dates))
        path[:, 0] = self.spot

        # Generate fractional Gaussian noise for all stocks
        for stock in range(self.nb_stocks):
            fracBM_noise[stock, :] = self.fBM.fgn()

        # Euler scheme
        for k in range(1, self.nb_dates + 1):
            previous_spots = path[:, k - 1]
            diffusion = self.diffusion_fct(previous_spots, k * self.dt)
            path[:, k] = (
                    previous_spots
                    + self.drift_fct(previous_spots, k * self.dt) * self.dt
                    + np.multiply(diffusion, fracBM_noise[:, k - 1]))

        return path


class FBMH1:
    """Fractional Brownian Motion for Hurst H=1 (deterministic linear trend)"""

    def __init__(self, n, length):
        self.n = n
        self.length = length

    def fbm(self):
        """Generate deterministic linear path"""
        return np.linspace(0, self.length, self.n + 1) * np.random.randn(1)


class FractionalBrownianMotion(Model):
    def __init__(self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
                 maturity, dividend=0, **keywords):
        super(FractionalBrownianMotion, self).__init__(
            drift=drift, dividend=dividend, volatility=volatility,
            nb_stocks=nb_stocks, nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, name="FractionalBrownianMotion"
        )
        self.hurst = hurst
        if self.hurst == 1:
            self.fBM = FBMH1(n=nb_dates, length=maturity)
        else:
            self.fBM = FBM(n=nb_dates, hurst=hurst, length=maturity, method='cholesky')
        self._nb_stocks = self.nb_stocks

    def _generate_one_path(self):
        """Returns a nparray (nb_stocks * nb_dates+1) with prices."""
        path = np.empty((self._nb_stocks, self.nb_dates + 1))
        for stock in range(self._nb_stocks):
            path[stock, :] = self.fBM.fbm() + self.spot
        return path

    def generate_one_path(self):
        return self._generate_one_path()


class FractionalBrownianMotionPathDep(FractionalBrownianMotion):
    """
    Path-dependent representation of fractional BM for non-Markovian algorithms.
    WARNING: This is specialized for 1D fractional BM experiments only.
    """

    def __init__(
            self, drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
            maturity, dividend=0, **keywords):
        assert nb_stocks == 1, "FractionalBrownianMotionPathDep only supports 1D"
        assert spot == 0, "FractionalBrownianMotionPathDep requires spot=0"
        super(FractionalBrownianMotionPathDep, self).__init__(
            drift, volatility, hurst, nb_paths, nb_stocks, nb_dates, spot,
            maturity, dividend=0, **keywords)
        self.nb_stocks = nb_dates + 1
        self._nb_stocks = 1

    def generate_one_path(self):
        """Returns path-dependent representation (nb_stocks=nb_dates+1, nb_dates+1)"""
        _path = self._generate_one_path()
        path = np.zeros((self.nb_stocks, self.nb_dates + 1))
        for i in range(self.nb_dates + 1):
            path[:i + 1, i] = np.flip(_path[0, :i + 1])
        return path, None


class Heston(Model):
    """
    Heston stochastic volatility model
    See: https://en.wikipedia.org/wiki/Heston_model

    Stock SDE: dS = mu*S*dt + sqrt(v)*S*dW
    Variance SDE: dv = -kappa*(v - v_bar)*dt + xi*sqrt(v)*dZ
    where dW and dZ have correlation rho
    """

    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, dividend=dividend, name="Heston"
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

        # Check Feller condition with tolerance for floating-point precision
        feller_lhs = 2 * self.speed * self.mean
        feller_rhs = self.volatility ** 2
        FELLER_TOLERANCE = 1e-10  # Tolerance for numerical precision

        # Only warn if Feller condition is violated beyond numerical precision
        if feller_lhs + FELLER_TOLERANCE < feller_rhs:
            warnings.warn(
                f"Feller condition not satisfied: 2*kappa*v_bar = "
                f"{feller_lhs:.6f} < xi^2 = {feller_rhs:.6f}. "
                f"Variance may become negative.",
                UserWarning
            )

    def drift_fct(self, x, t):
        del t
        return self.drift * x

    def diffusion_fct(self, x, t, v=0):
        del t
        v_positive = np.maximum(v, 0)
        return np.sqrt(v_positive) * x

    def var_drift_fct(self, x, v):
        """Variance drift: -kappa*(v - v_bar)"""
        v_positive = np.maximum(v, 0)
        return - self.speed * (np.subtract(v_positive, self.mean))

    def var_diffusion_fct(self, x, v):
        """Variance diffusion: xi*sqrt(v)"""
        v_positive = np.maximum(v, 0)
        return self.volatility * np.sqrt(v_positive)

    def generate_paths(self, start_X=None):
        """Generate paths using Euler-Maruyama scheme."""
        paths = np.empty((self.nb_paths, self.nb_stocks, self.nb_dates + 1))
        var_paths = np.empty((self.nb_paths, self.nb_stocks, self.nb_dates + 1))

        dt = self.maturity / self.nb_dates

        if start_X is not None:
            paths[:, :, 0] = start_X

        for i in range(self.nb_paths):
            if start_X is None:
                paths[i, :, 0] = self.spot
                var_paths[i, :, 0] = self.mean

            for k in range(1, self.nb_dates + 1):
                # Generate correlated Brownian increments
                normal_numbers_1 = np.random.normal(0, 1, self.nb_stocks)
                normal_numbers_2 = np.random.normal(0, 1, self.nb_stocks)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                # Update variance (Euler scheme)
                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + self.var_drift_fct(paths[i, :, k - 1],
                                             var_paths[i, :, k - 1]) * dt
                        + np.multiply(
                    self.var_diffusion_fct(paths[i, :, k - 1],
                                           var_paths[i, :, k - 1]), dZ))

                # Update stock price (Euler scheme)
                paths[i, :, k] = (
                        paths[i, :, k - 1]
                        + self.drift_fct(paths[i, :, k - 1], (k - 1) * dt) * dt
                        + np.multiply(self.diffusion_fct(paths[i, :, k - 1],
                                                         k * dt,
                                                         var_paths[i, :, k]), dW))

        return paths, var_paths

    def draw_path_heston(self, filename):
        """Plot a single Heston path showing stock price and variance."""
        nb_paths = self.nb_paths
        self.nb_paths = 1
        paths, var_paths = self.generate_paths()
        self.nb_paths = nb_paths

        one_path = paths[0, 0, :]
        one_var_path = var_paths[0, 0, :]
        dates = np.array([i for i in range(len(one_path))])

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color=color)
        ax1.plot(dates, one_path, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Variance', color=color)
        ax2.plot(dates, one_var_path, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.savefig(filename)
        plt.close()


class HestonWithVar(Heston):
    """Heston model that returns variance paths in generate_paths()"""

    def __init__(self, drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
                 nb_dates, spot, maturity, dividend=0., sine_coeff=None, **kwargs):
        super(HestonWithVar, self).__init__(
            drift, volatility, mean, speed, correlation, nb_stocks, nb_paths,
            nb_dates, spot, maturity, dividend=dividend, sine_coeff=sine_coeff,
            **kwargs
        )
        self.return_var = True


class RoughHeston(Model):
    """
    Rough Heston model with fractional variance process.
    Variance follows a fractional process with Hurst parameter H < 0.5.
    See: "Roughening Heston" paper

    Note: Computationally intensive due to fractional integration.
    """

    def __init__(self, drift, volatility, spot,
                 mean, speed, correlation,
                 nb_stocks, nb_paths, nb_dates, maturity,
                 nb_steps_mult=10, v0=None, hurst=0.25, dividend=0., **kwargs):
        super(RoughHeston, self).__init__(
            drift=drift, volatility=volatility, nb_stocks=nb_stocks,
            nb_paths=nb_paths, nb_dates=nb_dates,
            spot=spot, maturity=maturity, dividend=dividend,
            name="RoughHeston")
        self.mean = mean
        self.speed = speed
        self.nb_steps_mult = nb_steps_mult
        self.dt = self.maturity / (self.nb_dates * self.nb_steps_mult)
        self.correlation = correlation

        assert 0 < hurst < 0.5, "Rough Heston requires 0 < H < 0.5"
        self.H = hurst

        if v0 is None:
            self.v0 = self.mean
        else:
            self.v0 = v0

    def get_frac_var(self, vars, dZ, step, la, thet, vol):
        """
        Compute next fractional variance value using Euler scheme.
        See: "Roughening Heston" paper or https://github.com/sigurdroemer/rough_heston

        Args:
            vars: array with previous values of var process
            dZ: array with the BM increments for var process
            step: int > 0, the step of the integral
            la: lambda (mean reversion speed)
            thet: theta (long-run variance)
            vol: volatility of variance

        Returns:
            Next value of fractional var process
        """
        v0 = vars[0]
        times = (self.dt * step - np.linspace(0, self.dt * (step - 1), step)) ** \
                (self.H - 0.5)
        if len(vars.shape) == 2:
            times = np.repeat(np.expand_dims(times, 1), vars.shape[1], axis=1)
        int1 = np.sum(times * la * (thet - vars[:step]) * self.dt, axis=0)
        int2 = np.sum(times * vol * np.sqrt(vars[:step]) * dZ[:step], axis=0)
        v = v0 + (int1 + int2) / scispe.gamma(self.H + 0.5)
        return np.maximum(v, 0)

    def _generate_one_path(
            self, mu, la, thet, vol, start_X, nb_steps, v0=None):
        """Generate single path for one stock"""
        spot_path = np.empty((nb_steps + 1))
        spot_path[0] = start_X
        var_path = np.empty((nb_steps + 1))
        if v0 is None:
            var_path[0] = self.v0
        else:
            var_path[0] = v0

        # Log-normal dynamics for stock
        log_spot_drift = lambda v, t: (mu - 0.5 * np.maximum(v, 0))
        log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

        # Generate correlated Brownian increments
        normal_numbers_1 = np.random.normal(0, 1, nb_steps)
        normal_numbers_2 = np.random.normal(0, 1, nb_steps)
        dW = normal_numbers_1 * np.sqrt(self.dt)
        dZ = (self.correlation * normal_numbers_1 + np.sqrt(
            1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(self.dt)

        for k in range(1, nb_steps + 1):
            spot_path[k] = np.exp(
                np.log(spot_path[k - 1])
                + log_spot_drift(var_path[k - 1], (k - 1) * self.dt) * self.dt
                + log_spot_diffusion(var_path[k - 1]) * dW[k - 1]
            )
            var_path[k] = self.get_frac_var(var_path, dZ, k, la, thet, vol)

        return spot_path, var_path

    def _generate_paths(
            self, mu, la, thet, vol, start_X, nb_steps, v0=None, nb_stocks=1):
        """Generate multiple paths simultaneously (vectorized over stocks)"""
        spot_path = np.empty((nb_steps + 1, nb_stocks))
        spot_path[0] = start_X
        var_path = np.empty((nb_steps + 1, nb_stocks))
        if v0 is None:
            var_path[0] = self.v0
        else:
            var_path[0] = v0

        log_spot_drift = lambda v, t: (mu - 0.5 * np.maximum(v, 0))
        log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

        normal_numbers_1 = np.random.normal(0, 1, (nb_steps, nb_stocks))
        normal_numbers_2 = np.random.normal(0, 1, (nb_steps, nb_stocks))
        dW = normal_numbers_1 * np.sqrt(self.dt)
        dZ = (self.correlation * normal_numbers_1 + np.sqrt(
            1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(self.dt)

        for k in range(1, nb_steps + 1):
            spot_path[k] = np.exp(
                np.log(spot_path[k - 1])
                + log_spot_drift(var_path[k - 1], (k - 1) * self.dt) * self.dt
                + log_spot_diffusion(var_path[k - 1]) * dW[k - 1]
            )
            var_path[k] = self.get_frac_var(var_path, dZ, k, la, thet, vol)

        return spot_path, var_path

    def generate_one_path(self):
        """
        Generate paths under P measure for each dimension (stock).
        Uses fine time discretization (nb_dates * nb_steps_mult steps).
        """
        spot_paths = np.empty((self.nb_stocks, self.nb_dates + 1))
        for i in range(self.nb_stocks):
            spot_path, var_path = self._generate_one_path(
                self.drift, self.speed, self.mean, self.volatility,
                start_X=self.spot, nb_steps=self.nb_dates * self.nb_steps_mult)
            spot_paths[i, :] = spot_path[0::self.nb_steps_mult]

        return spot_paths

    def generate_paths(self, nb_paths=None):
        """Returns a nparray (nb_paths, nb_stocks, nb_dates+1) with prices."""
        nb_paths = nb_paths or self.nb_paths

        # Generate all paths simultaneously (treating each path*stock as separate)
        spot_paths, var_paths = self._generate_paths(
            self.drift, self.speed, self.mean, self.volatility,
            start_X=self.spot, nb_steps=self.nb_dates * self.nb_steps_mult,
            nb_stocks=self.nb_stocks * nb_paths
        )

        # Downsample to exercise dates only
        spot_paths = spot_paths[0::self.nb_steps_mult]
        var_paths = var_paths[0::self.nb_steps_mult]

        # Reshape to (nb_dates+1, nb_paths, nb_stocks)
        spot_paths = np.reshape(spot_paths,
                                (self.nb_dates + 1, nb_paths, self.nb_stocks))
        var_paths = np.reshape(var_paths,
                               (self.nb_dates + 1, nb_paths, self.nb_stocks))

        # Transpose to (nb_paths, nb_stocks, nb_dates+1)
        spot_paths = np.transpose(spot_paths, axes=(1, 2, 0))
        var_paths = np.transpose(var_paths, axes=(1, 2, 0))

        return spot_paths, var_paths


class RoughHestonWithVar(RoughHeston):
    """Rough Heston model that returns variance paths in generate_paths()"""

    def __init__(self, drift, volatility, spot,
                 mean, speed, correlation,
                 nb_stocks, nb_paths, nb_dates, maturity,
                 nb_steps_mult=10, v0=None, hurst=0.25, dividend=0., **kwargs):
        super(RoughHestonWithVar, self).__init__(
            drift, volatility, spot,
            mean, speed, correlation,
            nb_stocks, nb_paths, nb_dates, maturity,
            nb_steps_mult=nb_steps_mult, v0=v0, hurst=hurst, dividend=dividend,
            **kwargs
        )
        self.return_var = True


# ==============================================================================
# Dictionary for supported stock models to get them from their name

# Import RealDataModel (lazy import to avoid yfinance dependency if not used)
try:
    from optimal_stopping.data.real_data import RealDataModel
    _HAS_REAL_DATA = True
except ImportError:
    _HAS_REAL_DATA = False
    RealDataModel = None

STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    'FractionalBlackScholes': FractionalBlackScholes,
    'FractionalBrownianMotion': FractionalBrownianMotion,
    'FractionalBrownianMotionPathDep': FractionalBrownianMotionPathDep,
    "Heston": Heston,
    "RoughHeston": RoughHeston,
    "HestonWithVar": HestonWithVar,
    "RoughHestonWithVar": RoughHestonWithVar,
}

# Add RealData if available
if _HAS_REAL_DATA:
    STOCK_MODELS["RealData"] = RealDataModel


# ==============================================================================
# Dynamically register stored path models
# ==============================================================================

def _register_stored_models():
    """Scan stored_paths directory and register all stored models.

    This function is called automatically to make stored models available
    in STOCK_MODELS dictionary. Stored models are named like:
    'RealDataStored1700000000123', 'BlackScholesStored123', etc.
    """
    try:
        from optimal_stopping.data.stored_model import create_stored_model_class
        from optimal_stopping.data.path_storage import STORAGE_DIR
        from pathlib import Path
        import h5py

        if not STORAGE_DIR.exists():
            return  # No storage directory yet

        # Scan all .h5 files in storage directory
        for filepath in STORAGE_DIR.glob('*.h5'):
            try:
                # Read metadata to get model and storage_id
                with h5py.File(filepath, 'r') as f:
                    base_model = f.attrs.get('stock_model')
                    storage_id = f.attrs.get('storage_id')

                    if base_model and storage_id:
                        # Create and register the stored model class
                        model_name = f"{base_model}Stored{storage_id}"
                        STOCK_MODELS[model_name] = create_stored_model_class(base_model, storage_id)

            except Exception as e:
                # Silently skip corrupted files
                import warnings
                warnings.warn(f"Could not register stored model from {filepath.name}: {e}")
                continue

    except ImportError:
        # stored_model or path_storage not available - skip registration
        pass


# Register stored models on module import
_register_stored_models()

# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5, 'speed': 0.5, 'hurst': 0.05,
    'correlation': 0.5, 'nb_paths': 1, 'nb_dates': 100, 'maturity': 1.,
    'nb_stocks': 10, 'spot': 100}


def draw_stock_model(stock_model_name):
    """Utility function to visualize a stock model path"""
    hyperparam_test_stock_models['model_name'] = stock_model_name
    stockmodel = STOCK_MODELS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, _ = stockmodel.generate_paths()
    filename = '{}.pdf'.format(stock_model_name)

    # Draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    plt.plot(dates, one_path, label='stock path')
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # Example usage
    # draw_stock_model("BlackScholes")
    # draw_stock_model("FractionalBlackScholes")
    # heston = STOCK_MODELS["Heston"](**hyperparam_test_stock_models)
    # heston.draw_path_heston("heston.pdf")

    rHeston = RoughHeston(**hyperparam_test_stock_models)
    t = time.time()
    p = rHeston.generate_paths(1000)
    print("needed time: {}".format(time.time() - t))