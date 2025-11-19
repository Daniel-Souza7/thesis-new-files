"""
Barrier wrapper that applies barrier conditions to any base payoff.

Supports 11 barrier types:
- Single: UO, DO, UI, DI (4)
- Double: UODO, UIDI, UIDO, UODI (4)
- Custom: PTB, StepB, DStepB (3)
"""

import numpy as np
from .payoff import Payoff


class BarrierPayoff(Payoff):
    """Wrapper that adds barrier conditions to any base payoff."""

    is_path_dependent = True  # All barriers are path-dependent

    def __init__(self, base_payoff, barrier_type, strike, **kwargs):
        """
        Initialize barrier wrapper.

        Args:
            base_payoff: Base payoff class (NOT instance)
            barrier_type: One of UO, DO, UI, DI, UODO, UIDI, UIDO, UODI, PTB, StepB, DStepB
            strike: Strike price
            **kwargs: Barrier parameters (barrier, barrier_up, barrier_down, T1, T2, step_param1-4)
        """
        super().__init__(strike, **kwargs)
        self.base_payoff = base_payoff(strike, **kwargs)  # Instantiate base
        self.barrier_type = barrier_type.upper()

        # Extract barrier parameters
        self.barrier = kwargs.get('barrier', None)
        self.barrier_up = kwargs.get('barrier_up', None)
        self.barrier_down = kwargs.get('barrier_down', None)
        self.T1 = kwargs.get('T1', 0)  # Partial time barrier start
        self.T2 = kwargs.get('T2', None)  # Partial time barrier end (None = maturity)
        self.step_param1 = kwargs.get('step_param1', None)  # None means use risk-free rate
        self.step_param2 = kwargs.get('step_param2', None)
        self.step_param3 = kwargs.get('step_param3', None)
        self.step_param4 = kwargs.get('step_param4', None)

        # Model parameters for risk-free rate growth (default growth)
        self.rate = kwargs.get('rate', 0.05)  # Default 5% risk-free rate
        self.maturity = kwargs.get('maturity', 1.0)  # Default 1 year

        # Validate barrier type
        valid_types = ['UO', 'DO', 'UI', 'DI', 'UODO', 'UIDI', 'UIDO', 'UODI', 'PTB', 'STEPB', 'DSTEPB']
        if self.barrier_type not in valid_types:
            raise ValueError(f"Invalid barrier_type '{barrier_type}'. Must be one of {valid_types}")

    def eval(self, X):
        """
        Evaluate barrier payoff.

        Args:
            X: Array of shape (nb_paths, nb_stocks, nb_dates+1) - FULL path history

        Returns:
            Array of shape (nb_paths,) with payoff at maturity
        """
        nb_paths, nb_stocks, nb_dates = X.shape

        # Propagate initial prices to base payoff for normalization
        self.base_payoff.initial_prices = X[:, :, 0]

        # Evaluate base payoff at maturity
        if self.base_payoff.is_path_dependent:
            # Path-dependent payoffs need the full path history
            base_value = self.base_payoff.eval(X)
        else:
            # Standard payoffs only need the final price
            base_value = self.base_payoff.eval(X[:, :, -1])

        # Check barrier conditions
        barrier_active = self._check_barrier(X)

        # Apply barrier logic (active means payoff survives for Out, or activates for In)
        return base_value * barrier_active

    def _check_barrier(self, X):
        """
        Check if barrier conditions are met.

        Returns:
            Array of shape (nb_paths,) with 1.0 if barrier condition met, 0.0 otherwise
        """
        nb_paths, nb_stocks, nb_dates = X.shape

        if self.barrier_type in ['UO', 'DO', 'UI', 'DI']:
            return self._check_single_barrier(X)
        elif self.barrier_type in ['UODO', 'UIDI', 'UIDO', 'UODI']:
            return self._check_double_barrier(X)
        elif self.barrier_type == 'PTB':
            return self._check_partial_time_barrier(X)
        elif self.barrier_type == 'STEPB':
            return self._check_step_barrier(X)
        elif self.barrier_type == 'DSTEPB':
            return self._check_double_step_barrier(X)

    def _check_single_barrier(self, X):
        """Check single barrier conditions (UO, DO, UI, DI)."""
        nb_paths, nb_stocks, nb_dates = X.shape

        # Compute max/min over both time and assets
        max_over_path = np.max(X, axis=(1, 2))  # (nb_paths,)
        min_over_path = np.min(X, axis=(1, 2))  # (nb_paths,)

        if self.barrier_type == 'UO':  # Up-and-Out
            return (max_over_path < self.barrier).astype(float)
        elif self.barrier_type == 'DO':  # Down-and-Out
            return (min_over_path > self.barrier).astype(float)
        elif self.barrier_type == 'UI':  # Up-and-In
            return (max_over_path >= self.barrier).astype(float)
        elif self.barrier_type == 'DI':  # Down-and-In
            return (min_over_path <= self.barrier).astype(float)

    def _check_double_barrier(self, X):
        """Check double barrier conditions (UODO, UIDI, UIDO, UODI)."""
        max_over_path = np.max(X, axis=(1, 2))
        min_over_path = np.min(X, axis=(1, 2))

        if self.barrier_type == 'UODO':  # Double Knock-Out
            return ((min_over_path > self.barrier_down) & (max_over_path < self.barrier_up)).astype(float)
        elif self.barrier_type == 'UIDI':  # Double Knock-In
            return ((min_over_path <= self.barrier_down) | (max_over_path >= self.barrier_up)).astype(float)
        elif self.barrier_type == 'UIDO':  # Up-In-Down-Out
            return ((max_over_path >= self.barrier_up) & (min_over_path > self.barrier_down)).astype(float)
        elif self.barrier_type == 'UODI':  # Up-Out-Down-In
            return ((max_over_path < self.barrier_up) & (min_over_path <= self.barrier_down)).astype(float)

    def _check_partial_time_barrier(self, X):
        """Check partial time barrier (PTB)."""
        nb_paths, nb_stocks, nb_dates = X.shape

        # Determine time window
        t1_idx = int(self.T1 * nb_dates)
        t2_idx = int(self.T2 * nb_dates) if self.T2 else nb_dates

        # Extract portion of path within time window
        X_window = X[:, :, t1_idx:t2_idx]

        # Determine if call or put based on base payoff class name
        is_call = 'Call' in self.base_payoff.__class__.__name__

        if is_call:
            # Call → Up barrier logic
            max_in_window = np.max(X_window, axis=(1, 2))
            return (max_in_window < self.barrier).astype(float)
        else:
            # Put → Down barrier logic
            min_in_window = np.min(X_window, axis=(1, 2))
            return (min_in_window > self.barrier).astype(float)

    def _check_step_barrier(self, X):
        """
        Check step barrier (StepB) with time-varying barrier.

        By default, barrier grows at risk-free rate: B(t) = B(0) * exp(r * t * T / nb_dates)
        If step_param1 and step_param2 are provided, uses cumulative random walk instead.
        """
        nb_paths, nb_stocks, nb_dates = X.shape

        # Compute time-varying barrier
        if self.step_param1 is None or self.step_param2 is None:
            # Default: grow at risk-free rate
            # B(t) = B(0) * exp(rate * maturity * t / (nb_dates-1)) for t = 0, 1, ..., nb_dates-1
            time_steps = np.arange(nb_dates)  # Shape: (nb_dates,)
            # Use (nb_dates - 1) to ensure last step reaches maturity correctly
            growth_factor = np.exp(self.rate * self.maturity * time_steps / (nb_dates - 1))  # (nb_dates,)
            # Broadcast to (nb_paths, nb_dates)
            time_varying_barrier = self.barrier * growth_factor[np.newaxis, :]
        else:
            # User-specified: cumulative random walk
            random_increments = np.random.uniform(
                self.step_param1, self.step_param2,
                size=(nb_paths, nb_dates)
            )
            cumulative_walk = np.cumsum(random_increments, axis=1)
            time_varying_barrier = self.barrier + cumulative_walk  # (nb_paths, nb_dates)

        # Determine if call or put
        is_call = 'Call' in self.base_payoff.__class__.__name__

        if is_call:
            # Call → Check if ALL timesteps have max(S_i) < B(t)
            max_per_timestep = np.max(X, axis=1)  # (nb_paths, nb_dates)
            barrier_never_hit = np.all(max_per_timestep < time_varying_barrier, axis=1)
            return barrier_never_hit.astype(float)
        else:
            # Put → Check if ALL timesteps have min(S_i) > B(t)
            min_per_timestep = np.min(X, axis=1)  # (nb_paths, nb_dates)
            barrier_never_hit = np.all(min_per_timestep > time_varying_barrier, axis=1)
            return barrier_never_hit.astype(float)

    def _check_double_step_barrier(self, X):
        """
        Check double step barrier (DStepB) with two time-varying barriers.

        By default, both barriers grow at risk-free rate: B(t) = B(0) * exp(r * t * T / nb_dates)
        If step_param1-4 are provided, uses cumulative random walks instead.
        """
        nb_paths, nb_stocks, nb_dates = X.shape

        # Compute time-varying barriers
        if self.step_param1 is None or self.step_param2 is None:
            # Default: lower barrier grows at risk-free rate
            time_steps = np.arange(nb_dates)
            # Use (nb_dates - 1) to ensure last step reaches maturity correctly
            growth_factor = np.exp(self.rate * self.maturity * time_steps / (nb_dates - 1))
            barrier_lower = self.barrier_down * growth_factor[np.newaxis, :]  # (nb_paths, nb_dates)
        else:
            # User-specified: cumulative random walk
            random_increments_lower = np.random.uniform(
                self.step_param1, self.step_param2,
                size=(nb_paths, nb_dates)
            )
            cumulative_walk_lower = np.cumsum(random_increments_lower, axis=1)
            barrier_lower = self.barrier_down + cumulative_walk_lower

        if self.step_param3 is None or self.step_param4 is None:
            # Default: upper barrier grows at risk-free rate
            time_steps = np.arange(nb_dates)
            # Use (nb_dates - 1) to ensure last step reaches maturity correctly
            growth_factor = np.exp(self.rate * self.maturity * time_steps / (nb_dates - 1))
            barrier_upper = self.barrier_up * growth_factor[np.newaxis, :]  # (nb_paths, nb_dates)
        else:
            # User-specified: cumulative random walk
            random_increments_upper = np.random.uniform(
                self.step_param3, self.step_param4,
                size=(nb_paths, nb_dates)
            )
            cumulative_walk_upper = np.cumsum(random_increments_upper, axis=1)
            barrier_upper = self.barrier_up + cumulative_walk_upper

        # Check if stock stays within corridor at ALL timesteps
        max_per_timestep = np.max(X, axis=1)  # (nb_paths, nb_dates)
        min_per_timestep = np.min(X, axis=1)  # (nb_paths, nb_dates)

        within_corridor = np.all(
            (min_per_timestep > barrier_lower) & (max_per_timestep < barrier_upper),
            axis=1
        )

        return within_corridor.astype(float)


# Factory functions to create specific barrier payoffs
def create_barrier_payoff(base_payoff_class, barrier_type, name_suffix=""):
    """
    Factory to create a barrier payoff class.

    Args:
        base_payoff_class: Base payoff class
        barrier_type: Barrier type (UO, DO, UI, DI, etc.)
        name_suffix: Optional name suffix

    Returns:
        New class inheriting from BarrierPayoff
    """
    class_name = f"{barrier_type}_{base_payoff_class.__name__}{name_suffix}"

    def __init__(self, strike, **kwargs):
        BarrierPayoff.__init__(self, base_payoff_class, barrier_type, strike, **kwargs)

    # Create new class dynamically
    new_class = type(
        class_name,
        (BarrierPayoff,),
        {
            '__init__': __init__,
            'abbreviation': f"{barrier_type}-{base_payoff_class.abbreviation}",
            '__doc__': f"{barrier_type} barrier on {base_payoff_class.__name__}"
        }
    )

    return new_class
