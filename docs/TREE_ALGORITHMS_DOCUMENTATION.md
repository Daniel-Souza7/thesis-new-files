# Tree-Based Algorithms for American Option Pricing

## Table of Contents
1. [Introduction](#introduction)
2. [Cox-Ross-Rubinstein (CRR) Model](#cox-ross-rubinstein-crr-model)
3. [Leisen-Reimer (LR) Model](#leisen-reimer-lr-model)
4. [Trinomial Tree Model](#trinomial-tree-model)
5. [Implementation Methodology](#implementation-methodology)
6. [Alternative Approaches](#alternative-approaches)
7. [When to Use Each Algorithm](#when-to-use-each-algorithm)
8. [References](#references)

---

## Introduction

Tree-based methods (lattice methods) are fundamental approaches for pricing American options. Unlike Monte Carlo methods, tree methods work backwards from maturity to present, building a discrete lattice of possible stock price paths and determining the optimal exercise decision at each node.

### Key Advantages of Tree Methods
- **Exact optimal stopping**: Directly compute the exercise boundary
- **No train/eval split**: Deterministic computation on a lattice
- **Transparent decisions**: Clear visualization of when to exercise
- **Fast for single assets**: Efficient for low-dimensional problems

### Key Limitations
- **Single-asset only**: Exponential complexity for multi-asset options
- **Constant parameters**: Designed for constant volatility/rates (Black-Scholes)
- **Non-path-dependent**: Standard trees struggle with barriers and lookbacks
- **Convergence issues**: Some methods oscillate rather than converge smoothly

---

## Cox-Ross-Rubinstein (CRR) Model

### Method Overview

The CRR binomial tree is the most famous tree method for option pricing. At each time step, the stock can move **up** or **down** with probabilities calibrated to match the risk-neutral drift and volatility.

**Key Reference:**
> Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A simplified approach." *Journal of Financial Economics*, 7(3), 229-263.

### Mathematical Foundation

#### Tree Parameters

Given:
- $S_0$ = initial stock price
- $r$ = risk-free rate
- $\sigma$ = volatility
- $T$ = maturity
- $N$ = number of time steps
- $\Delta t = T / N$

The CRR model sets:

$$u = e^{\sigma \sqrt{\Delta t}}$$

$$d = \frac{1}{u} = e^{-\sigma \sqrt{\Delta t}}$$

$$p = \frac{e^{r \Delta t} - d}{u - d}$$

where:
- $u$ = up move factor
- $d$ = down move factor
- $p$ = risk-neutral probability of up move

#### Stock Price Evolution

At time step $i$ with $j$ up moves out of $i$ total moves:

$$S_{i,j} = S_0 \cdot u^j \cdot d^{i-j}$$

#### Backward Induction

Starting from maturity ($i = N$), work backwards to present ($i = 0$):

1. **At maturity**:
   $$V_{N,j} = \text{Payoff}(S_{N,j})$$

2. **At each earlier node**:
   $$C_{i,j} = e^{-r\Delta t} \left[ p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j} \right]$$
   $$V_{i,j} = \max\left( \text{Payoff}(S_{i,j}), C_{i,j} \right)$$

where $C_{i,j}$ is the continuation value and $V_{i,j}$ is the option value.

### Significance

- **Proves convergence**: CRR rigorously proved that binomial trees converge to the Black-Scholes formula for European options as $N \to \infty$
- **Risk-neutral framework**: Established the risk-neutral pricing paradigm
- **Simplicity**: Easy to understand and implement

### Known Issues: Oscillation

The CRR model suffers from **oscillating convergence**. As you increase $N$, the price doesn't smoothly approach the true value—it oscillates around it. This happens because:

1. The strike price $K$ may fall **between** tree nodes at maturity
2. Whether $K$ is above or below the nearest node changes as $N$ changes
3. This causes the price to jump up and down

**Example**: For an American put with $S_0 = 36$, $K = 40$, pricing with $N = 50$ might give \$4.52, but $N = 51$ gives \$4.44—even though both should be close to the true value of \$4.48.

---

## Leisen-Reimer (LR) Model

### Method Overview

The Leisen-Reimer model is an **improved binomial tree** that eliminates the oscillation problem of CRR. It uses **Peizer-Pratt inversion formulas** to ensure the tree nodes are centered on the strike price at maturity.

**Key Reference:**
> Leisen, D., & Reimer, M. (1996). "Binomial Models for Option Valuation - Examining and Improving Convergence." *Applied Mathematical Finance*, 3(4), 319-346.

### Mathematical Foundation

#### The Key Insight

Instead of arbitrarily choosing $u$ and $d$, LR uses the **Black-Scholes d₁ and d₂** to pre-calculate probabilities that align the tree with the strike:

$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

#### Peizer-Pratt Inversion

The Peizer-Pratt Method 2 formula approximates $\Phi(z)$ (cumulative normal) with high accuracy:

$$h(z, N) = \frac{1}{2} + \text{sign}(z') \sqrt{\frac{1}{4} - \frac{1}{4}e^{-\left(\frac{z'}{N + 1/3 + 0.1/(N+1)}\right)^2 \cdot (N + 1/6)}}$$

where $z' = z / \sqrt{N \Delta t / T}$.

We compute:
- $p_u = h(d_1, N)$ (probability matching)
- $p_d = h(d_2, N)$ (probability matching)

#### Tree Parameters

From these probabilities, solve for $u$, $d$, and $p$:

$$u = \frac{e^{r\Delta t} \cdot p_u}{p_d}$$

$$d = \frac{e^{r\Delta t} - p_d \cdot u}{1 - p_d}$$

$$p = p_d$$

#### Backward Induction

Same as CRR:

$$C_{i,j} = e^{-r\Delta t} \left[ p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j} \right]$$

$$V_{i,j} = \max\left( \text{Payoff}(S_{i,j}), C_{i,j} \right)$$

### Significance

- **Second-order convergence**: Achieves $O(N^{-2})$ convergence vs. $O(N^{-1})$ for CRR
- **Smooth convergence**: No oscillations—price smoothly approaches true value
- **Faster**: Can use fewer steps to achieve same accuracy as CRR
- **Strike alignment**: Guarantees strike price aligns with tree nodes at maturity

### Why It's Better

For the same $N$, LR is **significantly more accurate** than CRR:

| Method | N = 50 | N = 100 | N = 200 |
|--------|--------|---------|---------|
| CRR    | Oscillates | Oscillates | Oscillates |
| LR     | Smooth | Smooth | Smooth |

**Best Practice**: Use **odd** $N$ for LR (e.g., 51, 101, 201) for optimal Peizer-Pratt formula performance.

---

## Trinomial Tree Model

### Method Overview

The trinomial tree extends binomial trees by allowing **three** possible moves at each step: **up**, **middle** (stay flat), or **down**. This extra degree of freedom provides better stability and flexibility.

**Key Reference:**
> Boyle, P. P. (1986). "Option Valuation using a Three-Jump Process." *International Options Journal*, 3(3), 7-12.

### Mathematical Foundation

#### Tree Parameters (Boyle's Parameterization)

We use $\lambda = \sqrt{3}$ for stability. Given:
- $\Delta t = T / N$
- $\nu = r - \frac{1}{2}\sigma^2$ (drift in log-space)

Set:

$$u = e^{\sigma\sqrt{3\Delta t}}$$

$$d = \frac{1}{u}$$

$$m = 1 \quad \text{(middle branch: no change)}$$

#### Risk-Neutral Probabilities

Match the first two moments (mean and variance) of the risk-neutral distribution:

Let $\Delta x = \ln(u) = \sigma\sqrt{3\Delta t}$.

$$p_u = \frac{1}{2} \left[ \frac{\sigma^2 \Delta t + \nu^2 \Delta t^2}{(\Delta x)^2} + \frac{\nu \Delta t}{\Delta x} \right]$$

$$p_d = \frac{1}{2} \left[ \frac{\sigma^2 \Delta t + \nu^2 \Delta t^2}{(\Delta x)^2} - \frac{\nu \Delta t}{\Delta x} \right]$$

$$p_m = 1 - p_u - p_d$$

**Validity**: Require $0 \leq p_u, p_m, p_d \leq 1$. If violated, reduce $N$ or adjust parameters.

#### Stock Price Lattice

At time $i$, there are $2i + 1$ possible nodes. Node $j$ (ranging from $-i$ to $+i$) represents $j$ net up moves:

$$S_{i,j} = S_0 \cdot u^j$$

(where $j = \# \text{ups} - \# \text{downs}$)

#### Backward Induction

$$C_{i,j} = e^{-r\Delta t} \left[ p_u \cdot V_{i+1,j+1} + p_m \cdot V_{i+1,j} + p_d \cdot V_{i+1,j-1} \right]$$

$$V_{i,j} = \max\left( \text{Payoff}(S_{i,j}), C_{i,j} \right)$$

### Significance

- **Improved stability**: The middle branch reduces probability extremes
- **Flexibility**: Can handle time-dependent $\sigma$ and $r$ more easily
- **Recombining**: Still maintains a recombining lattice for efficiency
- **Numerical stability**: Less prone to negative probabilities than binomial for certain parameter ranges

### Advantages Over Binomial

1. **Better centering**: The middle branch keeps the tree centered on $S_0$ more naturally
2. **Time-varying parameters**: Easier to adapt $u$, $d$, $p_u$, $p_d$, $p_m$ at each time step
3. **Reduced oscillation**: Can be less prone to oscillation than CRR (though not as good as LR)

### Trade-offs

- **More nodes**: $O(N^2)$ nodes vs. $O(N^2)$ for binomial (same complexity, but higher constant factor)
- **More computations**: 3 branches vs. 2 per node (∼50% more computation)

---

## Implementation Methodology

### Our Specific Implementation Choices

#### 1. Default Number of Steps

**Choice**: We use $N = 50$ for CRR and Trinomial, $N = 51$ for LR.

**Justification**:
- **Balance**: 50 steps provides reasonable accuracy without excessive computation
- **Odd for LR**: 51 steps ensures Peizer-Pratt formulas work optimally
- **Testing baseline**: Can be increased for convergence studies

**Alternatives**:
- **Adaptive**: Start with small $N$, double until convergence
- **Richardson extrapolation**: Use $N$ and $2N$ to extrapolate the limit

#### 2. Single-Asset Restriction

**Choice**: Only support `nb_stocks = 1`.

**Justification**:
- **Dimensionality curse**: Multi-asset trees have $O(N^d)$ nodes for $d$ assets
- **Infeasible beyond d=2**: For 3+ assets, even $N=10$ is too slow
- **Monte Carlo better**: For baskets, use RLSM, RFQI, or LSM

**Alternatives**:
- **2-asset trees**: Could implement for $d=2$ with specialized data structures
- **Hybrid**: Use tree for single asset, MC for correlation
- **Sparse grids**: Advanced technique to reduce nodes in high dimensions

#### 3. Non-Path-Dependent Only

**Choice**: Raise error for `is_path_dependent = True` payoffs.

**Justification**:
- **Standard trees don't track history**: Only current node matters
- **Exploding state space**: Tracking full path history makes trees exponentially larger
- **Better alternatives exist**: Use SRLSM, SRFQI, or RRLSM for barriers/lookbacks

**Alternatives**:
- **Augmented state**: Add barrier status as extra dimension (still exponential)
- **Backward induction with memory**: Track whether barrier hit (works for some barriers)
- **PDE methods**: Finite differences handle barriers more naturally

#### 4. BlackScholes Model Assumption

**Choice**: Warn if model is not `BlackScholes`.

**Justification**:
- **Constant parameters**: Trees assume constant $\sigma$ and $r$
- **Heston incompatible**: Stochastic volatility requires variance state variable (2D tree)
- **Fractional/Rough**: Non-Markovian processes don't fit tree framework

**Alternatives**:
- **Average volatility**: For Heston, use $\bar{\sigma} = \sqrt{\theta}$ in tree
- **Implied tree**: Fit tree to match market prices (Derman-Kani, Dupire)
- **Variance as state**: Build 2D tree for $(S, v)$ in Heston

#### 5. Exercise Time Computation

**Choice**: Monte Carlo simulation on the tree's optimal stopping boundary.

**Justification**:
- **No analytical formula**: Trees give boundary, not expected exercise time
- **Simulation on lattice**: We simulate 10,000 paths on the tree and see when they exercise
- **Normalized to [0,1]**: Divide by $N$ for consistency with MC methods

**Alternatives**:
- **Analytical approximation**: For simple payoffs, use closed-form bounds
- **No exercise time**: Simply return `None` (trees don't naturally compute this)
- **Weighted average**: Weight each node by its probability (faster but less accurate)

#### 6. Backward Induction on Paths

**Choice**: Map continuous MC paths to discrete tree steps.

**Justification**:
- **API compatibility**: Match the interface of LSM, RLSM, etc.
- **Approximate**: Since paths don't align with tree nodes, we interpolate
- **For comparison**: Allows comparing tree policy to MC policy

**Alternatives**:
- **Raise error**: Don't support this method (trees don't need external paths)
- **Generate tree paths**: Return paths simulated on the tree itself
- **Nearest neighbor**: Match each path to closest tree node at each time

---

## Alternative Approaches

### When NOT to Use These Tree Implementations

#### 1. Multi-Asset Options (Baskets, Max-Calls, etc.)

**Problem**: Trees have $O(N^d)$ nodes for $d$ assets. For $d \geq 3$, this is infeasible.

**Better Alternatives**:
- **Monte Carlo**: RLSM, RFQI, LSM, FQI (linear in $d$)
- **Neural networks**: NLSM, DOS (handle high dimensions naturally)
- **Quasi-MC**: Low-discrepancy sequences for better convergence

**Example**: For a 5-asset MaxCall with $N=50$:
- Tree: $50^5 = 312,500,000$ nodes (impossible)
- RLSM: 10,000 paths × 50 steps = 500,000 evaluations (fast)

#### 2. Path-Dependent Payoffs (Barriers, Lookbacks, Asians)

**Problem**: Standard trees don't track path history. Adding history state explodes memory.

**Better Alternatives**:
- **Path-dependent MC**: SRLSM, SRFQI (designed for full paths)
- **Recurrent networks**: RRLSM (Echo State Networks remember history)
- **PDE methods**: Finite differences can handle some barriers elegantly

**Example**: For an Up-and-Out Barrier Call:
- Tree with barrier tracking: $O(N^2 \times 2)$ states (hit/not hit)
- SRLSM: Same $O(N \times M)$ complexity as standard SRLSM
- PDE: Natural boundary condition at barrier level

#### 3. Stochastic Volatility Models (Heston, Rough Heston)

**Problem**: Trees assume constant $\sigma$. Heston has stochastic $v_t$ (variance process).

**Better Alternatives**:
- **MC with full model**: RLSM on Heston paths (automatic)
- **2D tree**: Build $(S, v)$ lattice (complex but doable for 1 asset)
- **Fourier methods**: Characteristic functions (fast for European)

**Example**: For Heston model:
- 1D tree with average $\sigma$: Inaccurate (misses vol-of-vol)
- 2D tree $(S, v)$: $O(N^2)$ nodes but complex implementation
- RLSM with Heston paths: Works immediately, accurate

#### 4. High-Accuracy European Options

**Problem**: Trees converge slowly ($O(N^{-1})$ for CRR, $O(N^{-2})$ for LR).

**Better Alternatives**:
- **Black-Scholes formula**: Exact closed-form for calls/puts
- **Fourier methods**: Fast (FFT-based) for exotic Europeans
- **PDE solvers**: Finite differences with Crank-Nicolson (second-order accurate)

**Example**: European put under Black-Scholes:
- LR with $N=1000$: \$4.4782 (0.01s)
- Black-Scholes formula: \$4.4788 (0.0001s, exact)

#### 5. Exotic Time-Dependent Parameters

**Problem**: CRR and LR formulas assume constant $\sigma(t)$ and $r(t)$.

**Better Alternatives**:
- **Trinomial with time-varying**: Adjust $u_i, d_i, p_i$ at each step $i$
- **Implied trees**: Derman-Kani or Dupire local volatility trees
- **PDE with variable coefficients**: Finite differences naturally handle $\sigma(t)$

**Example**: For $\sigma(t) = 0.2 \cdot (1 + 0.5 \cos(2\pi t))$:
- CRR with average $\bar{\sigma}$: Inaccurate
- Trinomial with $u_i = f(\sigma(t_i))$: Accurate
- PDE: Most natural approach

---

### Advanced Tree Variants (Not Implemented Here)

#### 1. Implied Binomial Trees (Derman-Kani, Rubinstein)

**Idea**: Fit tree to match observed market prices of calls/puts at all strikes.

**Advantages**:
- **Model-free**: No need to assume Black-Scholes
- **Matches market**: By construction
- **Local volatility**: Extracts $\sigma(S, t)$ from prices

**Disadvantages**:
- **Requires market data**: Need full option chain
- **Unstable**: Small price errors cause large tree distortions
- **Complex**: Non-trivial calibration algorithm

**When to use**: Pricing exotic derivatives in a desk environment with live market data.

#### 2. Adaptive Mesh Refinement

**Idea**: Use fine grid near strike/barriers, coarse grid elsewhere.

**Advantages**:
- **Efficiency**: Fewer nodes for same accuracy
- **Targeted**: Focus resolution where needed

**Disadvantages**:
- **Non-recombining**: Loses recombining property (memory explosion)
- **Implementation complexity**: Bookkeeping for irregular grids

**When to use**: Research-grade pricing with strict accuracy requirements.

#### 3. Multi-Dimensional Trees (Quantized Grids)

**Idea**: For $d$-asset options, use sparse grid or quantization.

**Advantages**:
- **Handles multi-asset**: Can do $d=2, 3$ if sparse enough
- **Correlation**: Captures dependence between assets

**Disadvantages**:
- **Still exponential**: Only practical for $d \leq 3$
- **Algorithmic complexity**: Requires advanced data structures

**When to use**: When you need exact tree solution for 2-3 asset options.

---

## When to Use Each Algorithm

### Quick Decision Guide

| Scenario | Recommended Algorithm | Why? |
|----------|----------------------|------|
| **Single-asset American put/call** | **Leisen-Reimer (LR)** | Smooth convergence, fast, accurate |
| **Need fast rough estimate** | **CRR with small N (20-30)** | Simplest, good enough for quick valuation |
| **Time-varying σ(t) or r(t)** | **Trinomial** | Easy to adapt parameters per step |
| **Multi-asset (d ≥ 2)** | **RLSM, RFQI, LSM** | Trees infeasible; use Monte Carlo |
| **Barriers, lookbacks, Asians** | **SRLSM, SRFQI, RRLSM** | Path-dependent methods required |
| **Stochastic vol (Heston)** | **RLSM, NLSM** | MC handles stochastic processes naturally |
| **European option** | **Black-Scholes formula** | Closed-form is exact and instant |
| **Highest accuracy, single asset** | **LR with large N (>200)** | Best convergence properties |
| **Teaching/understanding** | **CRR** | Classic, intuitive, easy to visualize |

### Detailed Comparison

| Feature | CRR | LR | Trinomial |
|---------|-----|----|-----------|
| **Convergence rate** | $O(N^{-1})$ | $O(N^{-2})$ | $O(N^{-1})$ |
| **Oscillation** | Yes | No | Minimal |
| **Speed (same N)** | Fastest | Fast | Slower (3 branches) |
| **Accuracy (same N)** | Good | Excellent | Good |
| **Time-varying params** | Hard | Hard | Easy |
| **Stability** | Good | Excellent | Excellent |
| **Implementation** | Simple | Moderate | Moderate |
| **Best for** | Quick estimates | High accuracy | Flexible parameters |

---

## Practical Recommendations

### For Practitioners

1. **Default choice**: Use **Leisen-Reimer (LR)** with **N = 101 or 201** for production American option pricing.

2. **Quick checks**: Use **CRR with N = 50** for rapid prototyping and sanity checks.

3. **Parameter studies**: Use **Trinomial** if you need to vary $\sigma(t)$ or $r(t)$ in time.

4. **Convergence testing**: Always check convergence by running with $N, 2N, 4N$ and confirming prices stabilize.

5. **Multi-asset**: Switch to **RLSM** or **RFQI**—don't try to use trees for $d \geq 2$.

### For Researchers

1. **Benchmark**: Use **LR with N = 1000+** as a high-accuracy reference.

2. **Oscillation study**: Use **CRR** to demonstrate convergence issues in teaching.

3. **Extensions**: **Trinomial** is the best starting point for custom modifications (time-dependent, local vol, etc.).

4. **High-dimensional**: Explore **sparse grids** or **quantized trees** for $d=2, 3$.

---

## References

### Primary Literature

1. **Cox, J. C., Ross, S. A., & Rubinstein, M. (1979)**. "Option pricing: A simplified approach." *Journal of Financial Economics*, 7(3), 229-263.
   - **The CRR paper**: Established binomial trees and proved convergence to Black-Scholes.

2. **Leisen, D., & Reimer, M. (1996)**. "Binomial Models for Option Valuation - Examining and Improving Convergence." *Applied Mathematical Finance*, 3(4), 319-346.
   - **The LR paper**: Introduced Peizer-Pratt inversion for smooth second-order convergence.

3. **Boyle, P. P. (1986)**. "Option Valuation using a Three-Jump Process." *International Options Journal*, 3(3), 7-12.
   - **The Trinomial paper**: Extended binomial to trinomial for better flexibility.

### Additional Reading

4. **Hull, J. C. (2018)**. *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
   - Chapter 13: Binomial Trees (excellent textbook treatment)

5. **Glasserman, P. (2003)**. *Monte Carlo Methods in Financial Engineering*. Springer.
   - Chapter 8: Compares trees to MC methods

6. **Wilmott, P., Howison, S., & Dewynne, J. (1995)**. *The Mathematics of Financial Derivatives*. Cambridge University Press.
   - Chapter 3: Derives tree methods from PDEs

---

## Appendix: Implementation Details

### File Locations

- **CRR**: `optimal_stopping/algorithms/trees/crr.py`
- **Leisen-Reimer**: `optimal_stopping/algorithms/trees/leisen_reimer.py`
- **Trinomial**: `optimal_stopping/algorithms/trees/trinomial.py`

### Key Methods

All three classes implement the standard interface:

```python
class TreeAlgorithm:
    def __init__(self, model, payoff, n_steps=50, ...):
        """Initialize tree with model and payoff."""

    def price(self, train_eval_split=2):
        """Compute option price (returns price, comp_time)."""

    def get_exercise_time(self):
        """Return expected exercise time in [0, 1]."""

    def backward_induction_on_paths(self, stock_paths, var_paths=None):
        """Apply tree policy to Monte Carlo paths."""
```

### Registration

Algorithms are registered in:
- `optimal_stopping/run/run_algo.py` (main execution)
- `optimal_stopping/run/plot_convergence.py` (convergence studies)
- `optimal_stopping/run/create_video.py` (visualization)
- `frontend/api/pricing_engine.py` (web API)

As:
```python
_ALGOS = {
    "CRR": CRRTree,
    "LR": LeisenReimerTree,
    "Trinomial": TrinomialTree,
    ...
}
```

### Usage Example

```python
from optimal_stopping.data.stock_model import BlackScholes
from optimal_stopping.payoffs import Put
from optimal_stopping.algorithms.trees import LeisenReimerTree

# Setup
model = BlackScholes(spot=36, volatility=0.2, rate=0.06,
                     nb_stocks=1, maturity=1.0, nb_dates=50, nb_paths=10000)
payoff = Put(strike=40)

# Price with LR
algo = LeisenReimerTree(model, payoff, n_steps=101)
price, comp_time = algo.price()
print(f"American Put Price: ${price:.4f}")
```

---

## Conclusion

Tree-based methods remain fundamental tools for American option pricing, especially for single-asset problems. Our implementation provides three complementary approaches:

- **CRR**: The classic, simple, and intuitive method (with oscillation)
- **LR**: The modern, accurate, and recommended method (smooth convergence)
- **Trinomial**: The flexible method for time-varying parameters (extra stability)

For multi-asset, path-dependent, or stochastic volatility problems, use the Monte Carlo methods (RLSM, RFQI, SRLSM, SRFQI, RRLSM, etc.) instead.

**Final Recommendation**: Unless you have a specific reason to use CRR or Trinomial, **default to Leisen-Reimer (LR) with N = 101** for the best balance of accuracy, speed, and simplicity.

---

*Document version: 1.0*
*Last updated: 2025-12-06*
*Authors: Implementation based on Cox-Ross-Rubinstein (1979), Leisen-Reimer (1996), and Boyle (1986)*
