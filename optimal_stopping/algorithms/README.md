# Pricing Algorithms

This module implements pricing algorithms for American-style derivatives via optimal stopping. All algorithms estimate the continuation value function $c_n(x)$ to determine optimal exercise decisions.

## Mathematical Foundation

### The Optimal Stopping Problem

The value of an American option at time $n$ in state $x$ is given by the Snell envelope:

$$U_n(x) = \max\left\{ g(x), \mathbb{E}\left[\alpha U_{n+1}(X_{n+1}) \mid X_n = x\right] \right\}$$

where:
- $g(x)$: Payoff function (e.g., $(K - \bar{S})^+$ for a basket put)
- $\alpha = e^{-r\Delta t}$: One-period discount factor
- $c_n(x) = \mathbb{E}[\alpha U_{n+1}(X_{n+1}) \mid X_n = x]$: Continuation value

The optimal stopping rule exercises when $g(X_n) \geq c_n(X_n)$.

### Continuation Value Approximation

All regression-based methods approximate the continuation value as:

$$\hat{c}_n(x) = \sum_{k=1}^{K} \beta_{n,k} \psi_k(x) = \boldsymbol{\beta}_n^\top \boldsymbol{\psi}(x)$$

where $\boldsymbol{\psi}(x) \in \mathbb{R}^K$ is a vector of basis functions.

## Directory Structure

```
algorithms/
├── __init__.py          # Algorithm registry and exports
├── base.py              # Abstract base class
├── core/                # Core thesis algorithms
│   ├── rt.py            # RT (Randomized Thesis) - proposed
│   ├── rlsm.py          # Randomized LSM
│   ├── rfqi.py          # Randomized FQI
│   ├── lsm.py           # Classical LSM
│   ├── fqi.py           # Fitted Q-Iteration
│   └── eop.py           # European Option Price
├── deep/                # Deep learning baselines
│   ├── dos.py           # Deep Optimal Stopping
│   └── nlsm.py          # Neural LSM
├── recurrent/           # Path-dependent algorithms
│   ├── rrlsm.py         # Recurrent RLSM
│   ├── srlsm.py         # Special RLSM
│   └── srfqi.py         # Special RFQI
├── experimental/        # Research algorithms
│   ├── stochastic_mesh.py
│   ├── zap_q.py
│   └── dkl.py
└── utils/               # Shared utilities
    ├── neural_networks.py
    ├── randomized_neural_networks.py
    └── basis_functions.py
```

## Core Algorithms

### RT (Randomized Thesis Algorithm)

**File:** `core/rt.py`

The RT algorithm is the main contribution of this thesis. It uses randomized neural networks with the following enhancements:

1. **Dimension-adaptive neuron allocation** (Eq. 3.1 in thesis):
   $$K(d) = \max(2d, 5) \cdot \mathbf{1}_{1 \leq d \leq 9} + 1.5d \cdot \mathbf{1}_{10 \leq d \leq 49} + \ldots$$

2. **Random feature map**:
   $$\phi(x) = \left(\sigma(Wx + b)^\top, 1\right)^\top \in \mathbb{R}^K$$
   where $W \in \mathbb{R}^{(K-1) \times d}$ and $b \in \mathbb{R}^{K-1}$ are frozen random weights.

3. **Payoff augmentation**: Uses $\tilde{x}_n = (x^\top, g(x))^\top$ as input.

4. **Non-negativity constraint**: Enforces $\hat{c}_n(x) \geq 0$.

```python
from optimal_stopping.algorithms import RT

rt = RT(
    model=model,
    payoff=payoff,
    hidden_size=20,           # Or use adaptive sizing
    activation='leakyrelu',   # 'relu', 'tanh', 'elu', 'leakyrelu'
    use_payoff_as_input=True, # Payoff augmentation
    train_ITM_only=True       # Filter OTM paths
)
price, time = rt.price()
```

### RLSM (Randomized Least Squares Monte Carlo)

**File:** `core/rlsm.py`

The baseline randomized neural network method from Herrera et al. (2021). Uses fixed architecture (K=20 neurons) without RT's adaptive enhancements.

**Algorithm (backward induction):**

For $n = N-1, \ldots, 1$:
1. Compute features: $\phi(X_n^i)$ for all paths $i$
2. Solve regression: $\boldsymbol{\beta}_n = \alpha (\Phi^\top \Phi)^{-1} \Phi^\top \mathbf{p}_{n+1}$
3. Update payoffs: $p_n^i = g(X_n^i)$ if $g(X_n^i) \geq \hat{c}_n(X_n^i)$, else $\alpha p_{n+1}^i$

```python
from optimal_stopping.algorithms import RLSM

rlsm = RLSM(
    model=model,
    payoff=payoff,
    hidden_size=20,
    activation='leakyrelu'
)
price, time = rlsm.price()
```

### LSM (Least Squares Monte Carlo)

**File:** `core/lsm.py`

The classical method of Longstaff & Schwartz (2001) using polynomial basis functions.

**Basis functions:** Monomials up to degree $q$:
$$\psi(x) = \{1, x_1, x_2, \ldots, x_1^2, x_1 x_2, \ldots\}$$

The number of basis functions scales as $K = \binom{d+q}{q} = O(d^q)$, leading to the curse of dimensionality.

```python
from optimal_stopping.algorithms import LSM

lsm = LSM(model=model, payoff=payoff)
price, time = lsm.price()
```

### FQI (Fitted Q-Iteration)

**File:** `core/fqi.py`

Learns a single global Q-function $Q(n, x)$ incorporating time as an input, rather than separate functions per time step.

**Extended state:** $\tilde{x}_n = (n, N-n, x^\top)^\top \in \mathbb{R}^{d+2}$

**Iterative update:**
$$\boldsymbol{\beta}_{\ell+1} = \alpha \left(\sum_{n,i} \psi(n, X_n^i) \psi(n, X_n^i)^\top \right)^{-1} \sum_{n,i} \psi(n, X_n^i) p_{n+1}^i$$

```python
from optimal_stopping.algorithms import FQI

fqi = FQI(model=model, payoff=payoff, nb_epochs=30)
price, time = fqi.price()
```

### RFQI (Randomized Fitted Q-Iteration)

**File:** `core/rfqi.py`

Combines FQI with randomized features. **Note:** This algorithm was abandoned in the thesis due to memory scalability issues (see Section 3.1.3).

### EOP (European Option Price)

**File:** `core/eop.py`

Benchmark that exercises all paths at maturity (ignores early exercise). Used to validate algorithms when optimal exercise is at $T$ (e.g., calls with positive drift).

$$\hat{p}_{\text{EOP}} = \frac{1}{m} \sum_{i=1}^{m} e^{-rT} g(X_T^i)$$

```python
from optimal_stopping.algorithms import EOP

eop = EOP(model=model, payoff=payoff)
benchmark_price, time = eop.price()
```

## Deep Learning Algorithms

### DOS (Deep Optimal Stopping)

**File:** `deep/dos.py`

Uses fully trainable neural networks optimized via gradient descent (Becker et al., 2019).

**Limitations:**
- Non-convex optimization may converge to local minima
- No convergence guarantees
- Accuracy degrades in high dimensions (see Table 4.2)

```python
from optimal_stopping.algorithms import DOS

dos = DOS(model=model, payoff=payoff, nb_epochs=100)
price, time = dos.price()
```

### NLSM (Neural Least Squares Monte Carlo)

**File:** `deep/nlsm.py`

Deep learning variant of LSM replacing polynomial basis with trainable networks (Becker et al., 2020).

```python
from optimal_stopping.algorithms import NLSM

nlsm = NLSM(model=model, payoff=payoff, nb_epochs=100)
price, time = nlsm.price()
```

## Recurrent Algorithms

For path-dependent options (Asian, lookback, barriers), these algorithms use Echo State Networks (ESN) to process path history.

### RRLSM (Recurrent Randomized LSM)

**File:** `recurrent/rrlsm.py`

Uses ESN to encode path history $(S_0, \ldots, S_n)$ into hidden state $h_n$.

**Note:** The thesis (Section 3.1.4) demonstrates that RT's feedforward approach outperforms RRLSM by 12.1% on average.

```python
from optimal_stopping.algorithms import RRLSM

rrlsm = RRLSM(model=model, payoff=payoff)
price, time = rrlsm.price()
```

### SRLSM / SRFQI

**Files:** `recurrent/srlsm.py`, `recurrent/srfqi.py`

Special variants designed for path-dependent payoffs.

## Experimental Algorithms

Research algorithms not included in thesis benchmarks:

| Algorithm | File | Description |
|-----------|------|-------------|
| SM | `stochastic_mesh.py` | Stochastic Mesh (Broadie & Glasserman, 2004) |
| RSM1, RSM2 | `randomized_stochastic_mesh*.py` | Randomized mesh variants |
| ZAPQ | `zap_q.py` | Zap Q-learning |
| DKL | `dkl.py` | Deep Kernel Learning |

## Algorithm Registry

Algorithms are registered for dynamic loading:

```python
from optimal_stopping.algorithms import ALGORITHM_REGISTRY, get_algorithm

# Get algorithm by name
RT = get_algorithm('RT')

# List available algorithms
print(list(ALGORITHM_REGISTRY.keys()))
# ['RT', 'RLSM', 'RFQI', 'LSM', 'FQI', 'EOP', 'DOS', 'NLSM', 'RRLSM', 'SRLSM', 'SRFQI']
```

## Convergence Guarantees

**Theorem 2.1 (RLSM Convergence):** As $m \to \infty$ and $K \to \infty$, the price estimate $\hat{p}_0$ converges in probability to the true option price $U_0$.

**Key insight:** Randomized neural networks transform non-convex deep learning into convex linear regression, recovering the convergence guarantees of classical methods while achieving scalability comparable to deep networks.

## References

- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113-147.
- Herrera, C., Krach, F., Ruigrok, P., & Teichmann, J. (2021). Optimal stopping via randomized neural networks. *arXiv:2104.13669*.
- Becker, S., Cheridito, P., & Jentzen, A. (2019). Deep optimal stopping. *JMLR*, 20(74), 1-25.
- Tsitsiklis, J. N., & Van Roy, B. (2001). Regression methods for pricing complex American-style options. *IEEE Trans. Neural Networks*, 12(4), 694-703.
