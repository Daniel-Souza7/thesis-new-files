# New Payoffs Integration Summary

## Files Created:
1. `optimal_stopping/payoffs/niche.py` - 7 specialized niche payoffs
2. `optimal_stopping/payoffs/leverage.py` - 4 leveraged position payoffs  
3. Updated `optimal_stopping/payoffs/__init__.py` - exports all payoffs

## Files That Need Manual Updates:

### 1. `optimal_stopping/run/run_algo.py`
**Status:** Partially updated - imports and _PAYOFFS dict done, need to update:
- Add `k=None, notional=None, leverage=None, barrier_stop_loss=None` to `_run_algo()` signature (line 401)
- Update payoff instantiation logic (lines 416-432) to handle:
  - Niche payoffs: BestOfKCall, WorstOfKCall (need `k` parameter)
  - Leverage payoffs: All 4 (need `notional`, `leverage`, `barrier_stop_loss`)
- Update result dict to include new parameters (around line 585)
- Update call to _run_algo in _run_algos() to pass new params (around line 353)

### 2. `optimal_stopping/run/configs.py`
**Needs:** Add to _DefaultConfig (line 30):
```python
k: Iterable[int] = (2,)  # For niche payoffs
notional: Iterable[float] = (1.0,)  # For leverage payoffs
leverage: Iterable[float] = (2.0,)  # For leverage payoffs
barrier_stop_loss: Iterable[float] = (0.9,)  # For leverage stop-loss
```

### 3. Example config for new payoffs:
```python
niche_config = _FasterTable(
    algos=['RLSM', 'RFQI'],  # Non-path-dependent
    payoffs=['BestOfKCall', 'WorstOfKCall', 'RankWeightedBasketCall',
             'ChooserBasketOption', 'RangeCall', 'DispersionCall'],
    k=[2, 3, 5],  # For BestOfK/WorstOfK
    nb_stocks=[5, 10],
    strikes=[100],
    spots=[100],
    nb_paths=[10000],
    nb_dates=[9],
)

leverage_config = _FasterTable(
    algos=['RLSM'],  # Long/Short are non-path-dependent
    payoffs=['LeveragedBasketLongPosition', 'LeveragedBasketShortPosition'],
    notional=[1.0],
    leverage=[2.0, 3.0],
    nb_stocks=[5, 10],
    strikes=[100],
    spots=[100],
    nb_paths=[10000],
    nb_dates=[9],
)

leverage_stoploss_config = _FasterTable(
    algos=['SRLSM', 'SRFQI'],  # Stop-loss are PATH-DEPENDENT
    payoffs=['LeveragedBasketLongStopLoss', 'LeveragedBasketShortStopLoss'],
    notional=[1.0],
    leverage=[2.0],
    barrier_stop_loss=[0.9, 0.85],  # 10%, 15% loss limits
    nb_stocks=[5, 10],
    strikes=[100],
    spots=[100],
    nb_paths=[10000],
    nb_dates=[9],
)
```

## Payoff Details:

### Niche Payoffs (Non-path-dependent):
- **BestOfKCall**(strike, k=2): Average of top k stocks
- **WorstOfKCall**(strike, k=2): Average of bottom k stocks
- **RankWeightedBasketCall**(strike): Rank-weighted basket
- **ChooserBasketOption**(strike): Choose call or put
- **RangeCall**(strike): Max-Min spread
- **DispersionCall**(strike): Sum of deviations

### Leverage Payoffs:
- **LeveragedBasketLongPosition**(strike, notional=1.0, leverage=2.0): NON-path-dependent
- **LeveragedBasketShortPosition**(strike, notional=1.0, leverage=2.0): NON-path-dependent
- **LeveragedBasketLongStopLoss**(strike, notional=1.0, leverage=2.0, barrier_stop_loss=0.9): PATH-DEPENDENT
- **LeveragedBasketShortStopLoss**(strike, notional=1.0, leverage=2.0, barrier_stop_loss=1.1): PATH-DEPENDENT

## Next Steps for User:
1. Complete run_algo.py updates (payoff instantiation logic)
2. Update configs.py with new parameters
3. Test with simple config
4. Create comprehensive test configs
