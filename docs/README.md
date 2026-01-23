# Documentation

This directory contains documentation and reference materials for the `optimal_stopping` package.

## Contents

### `payoffs_index.tex`

LaTeX documentation containing mathematical definitions for all 360 payoff structures implemented in the framework. This includes:

- **30 Base Payoffs**: Mathematical formulas for basket options, single-asset options, Asian options, lookback options, and rank-based options
- **12 Barrier Conditions**: Definitions for knock-out, knock-in, double barriers, partial-time barriers, and step barriers
- **Combined Structures**: How barriers modify base payoffs

### Thesis Appendix

The LaTeX appendix for the thesis ("How to Run the Code") will be generated separately and can be found at:
- `appendix_howto.tex` (to be generated)

## Notation Conventions

All mathematical notation follows the thesis conventions:

| Symbol | Description |
|--------|-------------|
| $S_t$ | Asset price at time $t$ |
| $K$ | Strike price |
| $g(x)$ | Payoff function |
| $c_n(x)$ | Continuation value at time $n$, state $x$ |
| $\hat{c}_n(x)$ | Estimated continuation value |
| $\alpha$ | One-period discount factor $e^{-r\Delta t}$ |
| $\phi(x)$ | Random feature map |
| $\beta_n$ | Regression coefficients at time $n$ |

## Building Documentation

The LaTeX files can be compiled with:

```bash
cd docs
pdflatex payoffs_index.tex
```

For the full thesis integration, ensure the document class and bibliography are properly configured.
