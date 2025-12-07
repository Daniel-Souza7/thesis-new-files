"""Quick test to debug correlation in CRR tree."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import BasketPut
from optimal_stopping.algorithms.trees.crr import CRRTree

print("=" * 70)
print("Testing CRR with Correlation = 0.8")
print("=" * 70)

model = BlackScholes(
    spot=[100, 100],
    drift=0.05,
    volatility=[0.2, 0.2],
    rate=0.03,
    nb_stocks=2,
    nb_paths=100,
    nb_dates=50,
    maturity=1.0,
    correlation=0.8  # High correlation
)

print(f"\nModel correlation_matrix:\n{model.correlation_matrix}\n")

payoff = BasketPut(strike=100)

tree = CRRTree(model, payoff, n_steps=30)
price, time_taken = tree.price()

print(f"\nFinal Price: ${price:.4f}")
print(f"Time: {time_taken:.4f}s")
print()

print("=" * 70)
print("Testing CRR with Correlation = -0.8")
print("=" * 70)

model2 = BlackScholes(
    spot=[100, 100],
    drift=0.05,
    volatility=[0.2, 0.2],
    rate=0.03,
    nb_stocks=2,
    nb_paths=100,
    nb_dates=50,
    maturity=1.0,
    correlation=-0.8  # Negative correlation
)

print(f"\nModel correlation_matrix:\n{model2.correlation_matrix}\n")

tree2 = CRRTree(model2, payoff, n_steps=30)
price2, time_taken2 = tree2.price()

print(f"\nFinal Price: ${price2:.4f}")
print(f"Time: {time_taken2:.4f}s")
print()

print("=" * 70)
print("COMPARISON:")
print(f"  Correlation +0.8: ${price:.4f}")
print(f"  Correlation -0.8: ${price2:.4f}")
print(f"  Difference: ${abs(price - price2):.4f}")
print("=" * 70)
