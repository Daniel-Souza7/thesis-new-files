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
    """

    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Check if training should stop based on current score.

        Args:
            score: Current validation score
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            return False

        # Check for improvement based on mode
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            # Score improved
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            # No improvement
            self.counter += 1

        # Stop if patience exceeded
        should_stop = self.counter >= self.patience
        return should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
