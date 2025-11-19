# Payoff Registry System - Complete Catalog

This document catalogs all **360 unique payoffs** from the `optimal_stopping` Python codebase that have been implemented in the TypeScript frontend.

## Summary Statistics

- **Total Unique Payoffs**: 360
- **Base Payoffs**: 30
- **Barrier Variants**: 330 (30 base × 11 barrier types)
- **Barrier Types**: 11
- **Categories**: 4 (Single Asset, Basket, Barrier Single Asset, Barrier Basket)
- **Subcategories**: 6 (Simple, Asian, Lookback, Range, Dispersion, Rank)

## Complete Payoff List

### Single Asset Payoffs (12 base payoffs)

#### 1. Simple (2 payoffs)
- **Call** (`Call`) - European Call: max(0, S - K)
- **Put** (`Put`) - European Put: max(0, K - S)

#### 2. Lookback (4 payoffs)
- **LookbackFixedCall** (`LBFi-Call`) - Lookback Fixed Strike Call: max(0, max_over_time(S) - K)
- **LookbackFixedPut** (`LBFi-Put`) - Lookback Fixed Strike Put: max(0, K - min_over_time(S))
- **LookbackFloatCall** (`LBFl-Call`) - Lookback Floating Strike Call: max(0, S(T) - min_over_time(S))
- **LookbackFloatPut** (`LBFl-Put`) - Lookback Floating Strike Put: max(0, max_over_time(S) - S(T))

#### 3. Asian (4 payoffs)
- **AsianFixedStrikeCall_Single** (`AsianFi-Call`) - Asian Fixed Strike Call: max(0, avg_over_time(S) - K)
- **AsianFixedStrikePut_Single** (`AsianFi-Put`) - Asian Fixed Strike Put: max(0, K - avg_over_time(S))
- **AsianFloatingStrikeCall_Single** (`AsianFl-Call`) - Asian Floating Strike Call: max(0, S(T) - avg_over_time(S))
- **AsianFloatingStrikePut_Single** (`AsianFl-Put`) - Asian Floating Strike Put: max(0, avg_over_time(S) - S(T))

#### 4. Range (2 payoffs)
- **RangeCall_Single** (`Range-Call`) - Range Call: max(0, [max_over_time(S) - min_over_time(S)] - K)
- **RangePut_Single** (`Range-Put`) - Range Put: max(0, K - [max_over_time(S) - min_over_time(S)])

### Basket Payoffs (18 base payoffs)

#### 5. Simple (6 payoffs)
- **BasketCall** (`BskCall`) - Basket Call: max(0, mean(S) - K)
- **BasketPut** (`BskPut`) - Basket Put: max(0, K - mean(S))
- **GeometricCall** (`GeoCall`) - Geometric Call: max(0, geom_mean(S) - K)
- **GeometricPut** (`GeoPut`) - Geometric Put: max(0, K - geom_mean(S))
- **MaxCall** (`MaxCall`) - Max Call: max(0, max(S_i) - K)
- **MinPut** (`MinPut`) - Min Put: max(0, K - min(S_i))

#### 6. Asian (4 payoffs)
- **AsianFixedStrikeCall** (`AsianFi-BskCall`) - Asian Fixed Strike Basket Call: max(0, avg_over_time(mean(S)) - K)
- **AsianFixedStrikePut** (`AsianFi-BskPut`) - Asian Fixed Strike Basket Put: max(0, K - avg_over_time(mean(S)))
- **AsianFloatingStrikeCall** (`AsianFl-BskCall`) - Asian Floating Strike Basket Call: max(0, mean(S_T) - avg_over_time(mean(S)))
- **AsianFloatingStrikePut** (`AsianFl-BskPut`) - Asian Floating Strike Basket Put: max(0, avg_over_time(mean(S)) - mean(S_T))

#### 7. Dispersion (4 payoffs)
- **MaxDispersionCall** (`MaxDisp-BskCall`) - MaxDispersion Call: max(0, [max_i(S_i) - min_i(S_i)] - K)
- **MaxDispersionPut** (`MaxDisp-BskPut`) - MaxDispersion Put: max(0, K - [max_i(S_i) - min_i(S_i)])
- **DispersionCall** (`Disp-BskCall`) - Dispersion Call: max(0, σ(t) - K)
- **DispersionPut** (`Disp-BskPut`) - Dispersion Put: max(0, K - σ(t))

#### 8. Rank (4 payoffs)
- **BestOfKCall** (`BestK-BskCall`) - Best-of-K Basket Call: max(0, mean(top_k_prices) - K)
  - Parameters: `strike`, `k` (default: 2)
- **WorstOfKPut** (`WorstK-BskPut`) - Worst-of-K Basket Put: max(0, K - mean(bottom_k_prices))
  - Parameters: `strike`, `k` (default: 2)
- **RankWeightedBasketCall** (`Rank-BskCall`) - Rank-Weighted Basket Call: max(0, sum(w_i * S_(i)) - K)
  - Parameters: `strike`, `k` (default: 2), `weights` (optional)
- **RankWeightedBasketPut** (`Rank-BskPut`) - Rank-Weighted Basket Put: max(0, K - sum(w_i * S_(i)))
  - Parameters: `strike`, `k` (default: 2), `weights` (optional)

## Barrier Types (11 types)

Each of the 30 base payoffs can be combined with any of the following 11 barrier types to create 330 additional barrier variants:

### Single Barriers (4 types)
1. **UO** (Up-and-Out) - Option knocked out if price goes above barrier
2. **DO** (Down-and-Out) - Option knocked out if price goes below barrier
3. **UI** (Up-and-In) - Option activated if price goes above barrier
4. **DI** (Down-and-In) - Option activated if price goes below barrier

### Double Barriers (4 types)
5. **UODO** (Double Knock-Out) - Knocked out if price exits corridor
6. **UIDI** (Double Knock-In) - Activated if price exits corridor
7. **UIDO** (Up-In-Down-Out) - Activated by upper barrier, knocked out by lower
8. **UODI** (Up-Out-Down-In) - Knocked out by upper barrier, activated by lower

### Custom Barriers (3 types)
9. **PTB** (Partial Time Barrier) - Barrier only active during specified time window
10. **StepB** (Step Barrier) - Time-varying barrier (grows at risk-free rate or random walk)
11. **DStepB** (Double Step Barrier) - Two time-varying barriers (corridor)

## Barrier Parameters by Type

### Single Barriers (UO, DO, UI, DI)
- `barrier`: Barrier level

### Double Barriers (UODO, UIDI, UIDO, UODI)
- `barrier_up`: Upper barrier level
- `barrier_down`: Lower barrier level

### Partial Time Barrier (PTB)
- `barrier`: Barrier level
- `T1`: Start time (fraction of maturity, default: 0)
- `T2`: End time (fraction of maturity, None = maturity)

### Step Barrier (StepB)
- `barrier`: Initial barrier level B(0)
- `step_param1`: Random walk lower bound (None = use risk-free rate)
- `step_param2`: Random walk upper bound (None = use risk-free rate)

### Double Step Barrier (DStepB)
- `barrier_up`: Initial upper barrier B_up(0)
- `barrier_down`: Initial lower barrier B_down(0)
- `step_param1`: Lower barrier walk lower bound (None = risk-free rate)
- `step_param2`: Lower barrier walk upper bound (None = risk-free rate)
- `step_param3`: Upper barrier walk lower bound (None = risk-free rate)
- `step_param4`: Upper barrier walk upper bound (None = risk-free rate)

## Example Barrier Variants

For any base payoff (e.g., `BasketCall`), the following 11 barrier variants are automatically generated:

1. `UO_BasketCall` - Up-and-Out Basket Call
2. `DO_BasketCall` - Down-and-Out Basket Call
3. `UI_BasketCall` - Up-and-In Basket Call
4. `DI_BasketCall` - Down-and-In Basket Call
5. `UODO_BasketCall` - Double Knock-Out Basket Call
6. `UIDI_BasketCall` - Double Knock-In Basket Call
7. `UIDO_BasketCall` - Up-In-Down-Out Basket Call
8. `UODI_BasketCall` - Up-Out-Down-In Basket Call
9. `PTB_BasketCall` - Partial Time Barrier Basket Call
10. `StepB_BasketCall` - Step Barrier Basket Call
11. `DStepB_BasketCall` - Double Step Barrier Basket Call

## Path Dependency

### Standard (Non-Path-Dependent) Payoffs (10)
These depend only on the final stock prices at maturity:
- Call, Put
- BasketCall, BasketPut, GeometricCall, GeometricPut, MaxCall, MinPut
- DispersionCall, DispersionPut
- BestOfKCall, WorstOfKPut, RankWeightedBasketCall, RankWeightedBasketPut

### Path-Dependent Payoffs (20)
These require the full price history:
- **Lookback**: LookbackFixedCall, LookbackFixedPut, LookbackFloatCall, LookbackFloatPut
- **Asian**: AsianFixedStrikeCall_Single, AsianFixedStrikePut_Single, AsianFloatingStrikeCall_Single, AsianFloatingStrikePut_Single
- **Range**: RangeCall_Single, RangePut_Single
- **Basket Asian**: AsianFixedStrikeCall, AsianFixedStrikePut, AsianFloatingStrikeCall, AsianFloatingStrikePut
- **Basket Dispersion**: MaxDispersionCall, MaxDispersionPut

### All Barrier Variants (330)
All barrier variants are path-dependent by definition, as they must track whether the barrier condition was triggered during the option's lifetime.

## Files Created

1. **`/home/user/thesis-new-files/frontend/lib/payoffs.ts`**
   - TypeScript interface definitions
   - Complete catalog of all 360 payoffs
   - Helper functions for filtering and categorization
   - Barrier parameter definitions
   - Payoff statistics

2. **`/home/user/thesis-new-files/frontend/components/PayoffSelector.tsx`**
   - React component with retro arcade styling
   - Category dropdown (Single Asset, Basket, etc.)
   - Base payoff dropdown
   - Barrier type dropdown
   - Dynamic parameter inputs based on selected payoff
   - Real-time payoff summary display
   - Integrates with RetroPanel and RetroButton UI components

## Usage Example

```typescript
import { PayoffSelector } from '@/components/PayoffSelector';
import { getPayoffByName, PAYOFF_STATS } from '@/lib/payoffs';

// Use the selector component
<PayoffSelector
  onPayoffSelect={(payoff, parameters) => {
    console.log('Selected:', payoff.name);
    console.log('Parameters:', parameters);
  }}
  defaultCategory="Single Asset"
/>

// Access payoff data programmatically
const basketCall = getPayoffByName('BasketCall');
const uoBasketCall = getPayoffByName('UO_BasketCall');

console.log(PAYOFF_STATS);
// {
//   totalPayoffs: 360,
//   basePayoffs: 30,
//   barrierPayoffs: 330,
//   barrierTypes: 11,
//   categories: 4,
//   subcategories: 6
// }
```

## Implementation Notes

1. **Auto-Registration**: The Python codebase uses `__init_subclass__()` for automatic payoff registration. The TypeScript version generates all payoffs programmatically.

2. **Barrier Wrapper Pattern**: The Python codebase uses a single `BarrierPayoff` wrapper class that can wrap any base payoff. The TypeScript version represents this as a union of base payoff + barrier type.

3. **Parameter Validation**: The TypeScript interfaces define parameter requirements, but actual validation should be implemented in the backend.

4. **Naming Convention**: Barrier variants follow the pattern `{BarrierType}_{BasePayoffName}`, e.g., `UO_BasketCall`.

5. **Abbreviations**: Each payoff has a short abbreviation for display purposes, following the pattern `{BarrierType}-{BaseAbbreviation}`, e.g., `UO-BskCall`.

## Python Source Files

The TypeScript implementation mirrors the structure from these Python files:
- `/home/user/thesis-new-files/optimal_stopping/payoffs/__init__.py` - Registry system
- `/home/user/thesis-new-files/optimal_stopping/payoffs/payoff.py` - Base Payoff class
- `/home/user/thesis-new-files/optimal_stopping/payoffs/barrier_wrapper.py` - Barrier implementation
- `/home/user/thesis-new-files/optimal_stopping/payoffs/single_simple.py` - Call, Put
- `/home/user/thesis-new-files/optimal_stopping/payoffs/single_lookback.py` - Lookback options
- `/home/user/thesis-new-files/optimal_stopping/payoffs/single_asian.py` - Single Asian options
- `/home/user/thesis-new-files/optimal_stopping/payoffs/single_range.py` - Range options
- `/home/user/thesis-new-files/optimal_stopping/payoffs/basket_simple.py` - Basket, Geometric, Max, Min
- `/home/user/thesis-new-files/optimal_stopping/payoffs/basket_asian.py` - Basket Asian options
- `/home/user/thesis-new-files/optimal_stopping/payoffs/basket_range_dispersion.py` - Dispersion options
- `/home/user/thesis-new-files/optimal_stopping/payoffs/basket_rank.py` - Rank-based options
