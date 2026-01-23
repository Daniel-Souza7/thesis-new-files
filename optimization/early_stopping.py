"""
Early stopping callback for iterative algorithms (RFQI, SRFQI).

Monitors validation performance and stops training when no improvement is observed
for a specified number of epochs (patience).
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping callback to terminate training when validation performance plateaus.

    Usage:
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(max_epochs):
            val_score = train_and_validate()
            if early_stopping(val_score, epoch):
                print(f"Early stopping at epoch {epoch}")
                break

    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change in score to qualify as an improvement
        mode: 'max' to maximize score (default), 'min' to minimize
        divergence_threshold: Maximum realistic score (e.g., 5x spot price).
                            If score exceeds this, training diverged.
    """

    def __init__(self, patience=5, min_delta=0.001, mode='max', divergence_threshold=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.divergence_threshold = divergence_threshold

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.diverged = False

    def __call__(self, score, epoch):
        """
        Check if training should stop based on current score.

        Args:
            score: Current validation score
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check for divergence (NaN, Inf, or unrealistic values)
        if np.isnan(score) or np.isinf(score):
            print(f"  [ES] Epoch {epoch:3d}: ⚠️  DIVERGED - NaN or Inf detected!")
            self.diverged = True
            return True

        if self.divergence_threshold is not None and score > self.divergence_threshold:
            print(f"  [ES] Epoch {epoch:3d}: ⚠️  DIVERGED - Score {score:.2e} exceeds threshold {self.divergence_threshold:.2f}")
            self.diverged = True
            return True

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            print(f"  [ES] Epoch {epoch:3d}: Initial score = {score:.6f}")
            return False

        # Check for improvement based on mode
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            # Score improved
            delta = score - self.best_score
            print(f"  [ES] Epoch {epoch:3d}: ✓ IMPROVED! New best: {score:.6f} (Δ={delta:+.6f}) - Counter RESET")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            delta = score - self.best_score
            print(f"  [ES] Epoch {epoch:3d}: ✗ No improve. Counter: {self.counter}/{self.patience} (score={score:.6f}, best={self.best_score:.6f}, Δ={delta:+.6f})")

        # Stop if patience exceeded
        should_stop = self.counter >= self.patience
        if should_stop:
            print(f"  [ES] Epoch {epoch:3d}: STOPPING - Patience exhausted ({self.patience} epochs without improvement)")
        return should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
