# Payoff Quick Reference

## Complete Categorization of 360 Payoffs

```
TOTAL: 360 PAYOFFS
├── 30 Base Payoffs (no barriers)
└── 330 Barrier Variants (30 base × 11 barrier types)
```

## Base Payoffs by Category (30 total)

### 📊 SINGLE ASSET (12 payoffs)

```
Single Asset
├── Simple (2)
│   ├── Call
│   └── Put
├── Lookback (4)
│   ├── LookbackFixedCall
│   ├── LookbackFixedPut
│   ├── LookbackFloatCall
│   └── LookbackFloatPut
├── Asian (4)
│   ├── AsianFixedStrikeCall_Single
│   ├── AsianFixedStrikePut_Single
│   ├── AsianFloatingStrikeCall_Single
│   └── AsianFloatingStrikePut_Single
└── Range (2)
    ├── RangeCall_Single
    └── RangePut_Single
```

### 📈 BASKET (18 payoffs)

```
Basket
├── Simple (6)
│   ├── BasketCall
│   ├── BasketPut
│   ├── GeometricCall
│   ├── GeometricPut
│   ├── MaxCall
│   └── MinPut
├── Asian (4)
│   ├── AsianFixedStrikeCall
│   ├── AsianFixedStrikePut
│   ├── AsianFloatingStrikeCall
│   └── AsianFloatingStrikePut
├── Dispersion (4)
│   ├── MaxDispersionCall
│   ├── MaxDispersionPut
│   ├── DispersionCall
│   └── DispersionPut
└── Rank (4)
    ├── BestOfKCall
    ├── WorstOfKPut
    ├── RankWeightedBasketCall
    └── RankWeightedBasketPut
```

## Barrier Types (11 types)

Each of the 30 base payoffs has 11 barrier variants:

### 🚧 SINGLE BARRIERS (4 types)
```
1. UO  - Up-and-Out     (knocked out ↑)
2. DO  - Down-and-Out   (knocked out ↓)
3. UI  - Up-and-In      (activated ↑)
4. DI  - Down-and-In    (activated ↓)
```

### 🚧 DOUBLE BARRIERS (4 types)
```
5. UODO - Double Knock-Out        (knocked out if exits corridor)
6. UIDI - Double Knock-In         (activated if exits corridor)
7. UIDO - Up-In-Down-Out          (activated ↑, knocked out ↓)
8. UODI - Up-Out-Down-In          (knocked out ↑, activated ↓)
```

### 🚧 CUSTOM BARRIERS (3 types)
```
9.  PTB    - Partial Time Barrier   (active only during [T1, T2])
10. StepB  - Step Barrier           (time-varying barrier)
11. DStepB - Double Step Barrier    (time-varying corridor)
```

## Calculation

```
Base Payoffs:     30
Barrier Types:    11
Barrier Variants: 30 × 11 = 330
─────────────────────────────
TOTAL PAYOFFS:    360
```

## Breakdown by Path Dependency

### Non-Path-Dependent (10 base payoffs)
```
✓ Call, Put
✓ BasketCall, BasketPut, GeometricCall, GeometricPut, MaxCall, MinPut
✓ DispersionCall, DispersionPut
✓ BestOfKCall, WorstOfKPut, RankWeightedBasketCall, RankWeightedBasketPut
```

### Path-Dependent (20 base payoffs)
```
✓ All Lookback (4)
✓ All Asian Single (4)
✓ All Range Single (2)
✓ All Basket Asian (4)
✓ All Basket Dispersion MaxDisp* (2)
```

### All Barriers (330)
```
✓ ALL barrier variants are path-dependent
```

## Parameter Requirements

### Strike-Only Payoffs (26)
```
Parameters: [strike]

Single Asset Simple:    Call, Put
Single Asset Lookback:  LookbackFixed*, LookbackFloat*
Single Asset Asian:     AsianFixedStrike*_Single, AsianFloatingStrike*_Single
Single Asset Range:     Range*_Single
Basket Simple:          BasketCall, BasketPut, GeometricCall, GeometricPut, MaxCall, MinPut
Basket Asian:           AsianFixedStrike*, AsianFloatingStrike*
Basket Dispersion:      MaxDispersion*, Dispersion*
```

### Rank Payoffs with k (4)
```
Parameters: [strike, k]

BestOfKCall, WorstOfKPut
RankWeightedBasketCall, RankWeightedBasketPut (also accepts weights[])
```

## Naming Convention

### Base Payoffs
```
Format: {PayoffName}
Example: BasketCall
```

### Barrier Payoffs
```
Format: {BarrierType}_{PayoffName}
Example: UO_BasketCall
```

### Abbreviations
```
Format: {BarrierType}-{PayoffAbbrev}
Example: UO-BskCall
```

## Common Use Cases

### European Options (Single Asset)
```typescript
getPayoffByName('Call')
getPayoffByName('Put')
```

### Barrier European Options
```typescript
getPayoffByName('UO_Call')   // Up-and-Out Call
getPayoffByName('DO_Put')    // Down-and-Out Put
```

### Basket Options
```typescript
getPayoffByName('BasketCall')
getPayoffByName('MaxCall')
getPayoffByName('MinPut')
```

### Path-Dependent Options
```typescript
getPayoffByName('LookbackFixedCall')
getPayoffByName('AsianFixedStrikeCall_Single')
getPayoffByName('RangeCall_Single')
```

### Complex Basket Options
```typescript
getPayoffByName('BestOfKCall')           // Top k performers
getPayoffByName('RankWeightedBasketCall') // Custom weighted
getPayoffByName('MaxDispersionCall')     // Dispersion trading
```

### Step Barrier Options
```typescript
getPayoffByName('StepB_BasketCall')   // Time-varying barrier
getPayoffByName('DStepB_MaxCall')     // Time-varying corridor
```

## File Locations

### TypeScript Files
```
/home/user/thesis-new-files/frontend/
├── lib/
│   └── payoffs.ts                    # Payoff registry and types
├── components/
│   └── PayoffSelector.tsx            # Payoff selector component
├── PAYOFF_REGISTRY_SUMMARY.md        # Detailed documentation
└── PAYOFF_QUICK_REFERENCE.md         # This file
```

### Python Source Files
```
/home/user/thesis-new-files/optimal_stopping/payoffs/
├── __init__.py                       # Registry system
├── payoff.py                         # Base class
├── barrier_wrapper.py                # Barrier implementation
├── single_simple.py                  # Call, Put
├── single_lookback.py                # Lookback options
├── single_asian.py                   # Single Asian options
├── single_range.py                   # Range options
├── basket_simple.py                  # Basket, Geometric, Max, Min
├── basket_asian.py                   # Basket Asian options
├── basket_range_dispersion.py        # Dispersion options
└── basket_rank.py                    # Rank-based options
```

## TypeScript API Examples

### Get All Payoffs by Category
```typescript
import { getPayoffsByCategory } from '@/lib/payoffs';

const singleAsset = getPayoffsByCategory('Single Asset');
const basket = getPayoffsByCategory('Basket');
```

### Get Base Payoffs Only
```typescript
import { getBasePayoffs } from '@/lib/payoffs';

const basePayoffs = getBasePayoffs(); // 30 payoffs
```

### Get Barrier Payoffs Only
```typescript
import { getBarrierPayoffs } from '@/lib/payoffs';

const barrierPayoffs = getBarrierPayoffs(); // 330 payoffs
```

### Get Barrier Parameters
```typescript
import { getBarrierParameters } from '@/lib/payoffs';

const uoParams = getBarrierParameters('UO');
// [{ name: 'barrier', type: 'number', required: true, ... }]

const stepBParams = getBarrierParameters('StepB');
// [
//   { name: 'barrier', type: 'number', required: true, ... },
//   { name: 'step_param1', type: 'number', required: false, ... },
//   { name: 'step_param2', type: 'number', required: false, ... }
// ]
```

### Check Payoff Statistics
```typescript
import { PAYOFF_STATS } from '@/lib/payoffs';

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
