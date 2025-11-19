/**
 * Payoff Registry System
 *
 * Catalogs all 360 payoffs from the optimal_stopping Python codebase:
 * - 30 base payoffs
 * - 330 barrier variants (30 base × 11 barrier types)
 */

export interface PayoffInfo {
  name: string;
  abbreviation: string;
  category: PayoffCategory;
  subcategory?: string;
  isPathDependent: boolean;
  requiresMultipleAssets: boolean;
  description: string;
  parameters: PayoffParameter[];
  barrierType?: BarrierType;
}

export interface PayoffParameter {
  name: string;
  type: 'number' | 'number[]' | 'integer';
  required: boolean;
  default?: any;
  description: string;
  min?: number;
  max?: number;
}

export type PayoffCategory =
  | 'Single Asset'
  | 'Basket'
  | 'Barrier Single Asset'
  | 'Barrier Basket';

export type PayoffSubcategory =
  | 'Simple'
  | 'Asian'
  | 'Lookback'
  | 'Range'
  | 'Dispersion'
  | 'Rank';

export type BarrierType =
  | 'None'
  | 'UO'   // Up-and-Out
  | 'DO'   // Down-and-Out
  | 'UI'   // Up-and-In
  | 'DI'   // Down-and-In
  | 'UODO' // Double Knock-Out
  | 'UIDI' // Double Knock-In
  | 'UIDO' // Up-In-Down-Out
  | 'UODI' // Up-Out-Down-In
  | 'PTB'  // Partial Time Barrier
  | 'StepB'  // Step Barrier
  | 'DStepB'; // Double Step Barrier

/**
 * Base payoff definitions (30 total)
 */
const BASE_PAYOFFS: Omit<PayoffInfo, 'barrierType'>[] = [
  // ============================================================
  // SINGLE ASSET SIMPLE (2)
  // ============================================================
  {
    name: 'Call',
    abbreviation: 'Call',
    category: 'Single Asset',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: false,
    description: 'European Call: max(0, S - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'Put',
    abbreviation: 'Put',
    category: 'Single Asset',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: false,
    description: 'European Put: max(0, K - S)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // SINGLE ASSET LOOKBACK (4)
  // ============================================================
  {
    name: 'LookbackFixedCall',
    abbreviation: 'LBFi-Call',
    category: 'Single Asset',
    subcategory: 'Lookback',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Lookback Fixed Strike Call: max(0, max_over_time(S) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'LookbackFixedPut',
    abbreviation: 'LBFi-Put',
    category: 'Single Asset',
    subcategory: 'Lookback',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Lookback Fixed Strike Put: max(0, K - min_over_time(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'LookbackFloatCall',
    abbreviation: 'LBFl-Call',
    category: 'Single Asset',
    subcategory: 'Lookback',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Lookback Floating Strike Call: max(0, S(T) - min_over_time(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'LookbackFloatPut',
    abbreviation: 'LBFl-Put',
    category: 'Single Asset',
    subcategory: 'Lookback',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Lookback Floating Strike Put: max(0, max_over_time(S) - S(T))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // SINGLE ASSET ASIAN (4)
  // ============================================================
  {
    name: 'AsianFixedStrikeCall_Single',
    abbreviation: 'AsianFi-Call',
    category: 'Single Asset',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Asian Fixed Strike Call (single): max(0, avg_over_time(S) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFixedStrikePut_Single',
    abbreviation: 'AsianFi-Put',
    category: 'Single Asset',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Asian Fixed Strike Put (single): max(0, K - avg_over_time(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFloatingStrikeCall_Single',
    abbreviation: 'AsianFl-Call',
    category: 'Single Asset',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Asian Floating Strike Call (single): max(0, S(T) - avg_over_time(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFloatingStrikePut_Single',
    abbreviation: 'AsianFl-Put',
    category: 'Single Asset',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Asian Floating Strike Put (single): max(0, avg_over_time(S) - S(T))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // SINGLE ASSET RANGE (2)
  // ============================================================
  {
    name: 'RangeCall_Single',
    abbreviation: 'Range-Call',
    category: 'Single Asset',
    subcategory: 'Range',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Range Call (single): max(0, [max_over_time(S) - min_over_time(S)] - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'RangePut_Single',
    abbreviation: 'Range-Put',
    category: 'Single Asset',
    subcategory: 'Range',
    isPathDependent: true,
    requiresMultipleAssets: false,
    description: 'Range Put (single): max(0, K - [max_over_time(S) - min_over_time(S)])',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // BASKET SIMPLE (6)
  // ============================================================
  {
    name: 'BasketCall',
    abbreviation: 'BskCall',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Basket Call: max(0, mean(S) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'BasketPut',
    abbreviation: 'BskPut',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Basket Put: max(0, K - mean(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'GeometricCall',
    abbreviation: 'GeoCall',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Geometric Call: max(0, geom_mean(S) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'GeometricPut',
    abbreviation: 'GeoPut',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Geometric Put: max(0, K - geom_mean(S))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'MaxCall',
    abbreviation: 'MaxCall',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Max Call: max(0, max(S_i) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'MinPut',
    abbreviation: 'MinPut',
    category: 'Basket',
    subcategory: 'Simple',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Min Put: max(0, K - min(S_i))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // BASKET ASIAN (4)
  // ============================================================
  {
    name: 'AsianFixedStrikeCall',
    abbreviation: 'AsianFi-BskCall',
    category: 'Basket',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'Asian Fixed Strike Basket Call: max(0, avg_over_time(mean(S)) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFixedStrikePut',
    abbreviation: 'AsianFi-BskPut',
    category: 'Basket',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'Asian Fixed Strike Basket Put: max(0, K - avg_over_time(mean(S)))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFloatingStrikeCall',
    abbreviation: 'AsianFl-BskCall',
    category: 'Basket',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'Asian Floating Strike Basket Call: max(0, mean(S_T) - avg_over_time(mean(S)))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'AsianFloatingStrikePut',
    abbreviation: 'AsianFl-BskPut',
    category: 'Basket',
    subcategory: 'Asian',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'Asian Floating Strike Basket Put: max(0, avg_over_time(mean(S)) - mean(S_T))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // BASKET DISPERSION (4)
  // ============================================================
  {
    name: 'MaxDispersionCall',
    abbreviation: 'MaxDisp-BskCall',
    category: 'Basket',
    subcategory: 'Dispersion',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'MaxDispersion Call: max(0, [max_i(S_i) - min_i(S_i)] - K) over all stocks and time',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'MaxDispersionPut',
    abbreviation: 'MaxDisp-BskPut',
    category: 'Basket',
    subcategory: 'Dispersion',
    isPathDependent: true,
    requiresMultipleAssets: true,
    description: 'MaxDispersion Put: max(0, K - [max_i(S_i) - min_i(S_i)]) over all stocks and time',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'DispersionCall',
    abbreviation: 'Disp-BskCall',
    category: 'Basket',
    subcategory: 'Dispersion',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Dispersion Call: max(0, σ(t) - K) where σ is std dev of current prices',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },
  {
    name: 'DispersionPut',
    abbreviation: 'Disp-BskPut',
    category: 'Basket',
    subcategory: 'Dispersion',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Dispersion Put: max(0, K - σ(t)) where σ is std dev of current prices',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' }
    ]
  },

  // ============================================================
  // BASKET RANK (4)
  // ============================================================
  {
    name: 'BestOfKCall',
    abbreviation: 'BestK-BskCall',
    category: 'Basket',
    subcategory: 'Rank',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Best-of-K Basket Call: max(0, mean(top_k_prices) - K)',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' },
      { name: 'k', type: 'integer', required: false, default: 2, description: 'Number of top performers', min: 1 }
    ]
  },
  {
    name: 'WorstOfKPut',
    abbreviation: 'WorstK-BskPut',
    category: 'Basket',
    subcategory: 'Rank',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Worst-of-K Basket Put: max(0, K - mean(bottom_k_prices))',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' },
      { name: 'k', type: 'integer', required: false, default: 2, description: 'Number of bottom performers', min: 1 }
    ]
  },
  {
    name: 'RankWeightedBasketCall',
    abbreviation: 'Rank-BskCall',
    category: 'Basket',
    subcategory: 'Rank',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Rank-Weighted Basket Call: max(0, sum(w_i * S_(i)) - K) for top k performers',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' },
      { name: 'k', type: 'integer', required: false, default: 2, description: 'Number of top performers', min: 1 },
      { name: 'weights', type: 'number[]', required: false, description: 'Custom weights (must sum to 1)' }
    ]
  },
  {
    name: 'RankWeightedBasketPut',
    abbreviation: 'Rank-BskPut',
    category: 'Basket',
    subcategory: 'Rank',
    isPathDependent: false,
    requiresMultipleAssets: true,
    description: 'Rank-Weighted Basket Put: max(0, K - sum(w_i * S_(i))) for top k performers',
    parameters: [
      { name: 'strike', type: 'number', required: true, description: 'Strike price K' },
      { name: 'k', type: 'integer', required: false, default: 2, description: 'Number of top performers', min: 1 },
      { name: 'weights', type: 'number[]', required: false, description: 'Custom weights (must sum to 1)' }
    ]
  }
];

/**
 * All barrier types (11 total)
 */
export const BARRIER_TYPES: BarrierType[] = [
  'None',
  'UO',    // Up-and-Out
  'DO',    // Down-and-Out
  'UI',    // Up-and-In
  'DI',    // Down-and-In
  'UODO',  // Double Knock-Out
  'UIDI',  // Double Knock-In
  'UIDO',  // Up-In-Down-Out
  'UODI',  // Up-Out-Down-In
  'PTB',   // Partial Time Barrier
  'StepB', // Step Barrier
  'DStepB' // Double Step Barrier
];

/**
 * Barrier parameter definitions by type
 */
export function getBarrierParameters(barrierType: BarrierType): PayoffParameter[] {
  switch (barrierType) {
    case 'None':
      return [];

    case 'UO':
    case 'DO':
    case 'UI':
    case 'DI':
      return [
        { name: 'barrier', type: 'number', required: true, description: 'Barrier level' }
      ];

    case 'UODO':
    case 'UIDI':
    case 'UIDO':
    case 'UODI':
      return [
        { name: 'barrier_up', type: 'number', required: true, description: 'Upper barrier level' },
        { name: 'barrier_down', type: 'number', required: true, description: 'Lower barrier level' }
      ];

    case 'PTB':
      return [
        { name: 'barrier', type: 'number', required: true, description: 'Barrier level' },
        { name: 'T1', type: 'number', required: false, default: 0, description: 'Start time (fraction of maturity)', min: 0, max: 1 },
        { name: 'T2', type: 'number', required: false, description: 'End time (fraction of maturity, None = maturity)', min: 0, max: 1 }
      ];

    case 'StepB':
      return [
        { name: 'barrier', type: 'number', required: true, description: 'Initial barrier level B(0)' },
        { name: 'step_param1', type: 'number', required: false, description: 'Random walk lower bound (None = use risk-free rate)' },
        { name: 'step_param2', type: 'number', required: false, description: 'Random walk upper bound (None = use risk-free rate)' }
      ];

    case 'DStepB':
      return [
        { name: 'barrier_up', type: 'number', required: true, description: 'Initial upper barrier B_up(0)' },
        { name: 'barrier_down', type: 'number', required: true, description: 'Initial lower barrier B_down(0)' },
        { name: 'step_param1', type: 'number', required: false, description: 'Lower barrier walk lower bound (None = risk-free rate)' },
        { name: 'step_param2', type: 'number', required: false, description: 'Lower barrier walk upper bound (None = risk-free rate)' },
        { name: 'step_param3', type: 'number', required: false, description: 'Upper barrier walk lower bound (None = risk-free rate)' },
        { name: 'step_param4', type: 'number', required: false, description: 'Upper barrier walk upper bound (None = risk-free rate)' }
      ];

    default:
      return [];
  }
}

/**
 * Generate all 360 payoffs (30 base + 330 barrier variants)
 */
function generateAllPayoffs(): PayoffInfo[] {
  const allPayoffs: PayoffInfo[] = [];

  // Add base payoffs with barrierType = 'None'
  for (const basePayoff of BASE_PAYOFFS) {
    allPayoffs.push({
      ...basePayoff,
      barrierType: 'None'
    });
  }

  // Generate barrier variants for each base payoff
  for (const basePayoff of BASE_PAYOFFS) {
    for (const barrierType of BARRIER_TYPES) {
      if (barrierType === 'None') continue; // Skip 'None' as we already added base payoffs

      const barrierPayoff: PayoffInfo = {
        name: `${barrierType}_${basePayoff.name}`,
        abbreviation: `${barrierType}-${basePayoff.abbreviation}`,
        category: basePayoff.category.includes('Basket') ? 'Barrier Basket' : 'Barrier Single Asset',
        subcategory: basePayoff.subcategory,
        isPathDependent: true, // All barriers are path-dependent
        requiresMultipleAssets: basePayoff.requiresMultipleAssets,
        description: `${barrierType} barrier on ${basePayoff.description}`,
        parameters: [
          ...basePayoff.parameters,
          ...getBarrierParameters(barrierType)
        ],
        barrierType
      };

      allPayoffs.push(barrierPayoff);
    }
  }

  return allPayoffs;
}

/**
 * All 360 payoffs
 */
export const ALL_PAYOFFS = generateAllPayoffs();

/**
 * Get payoffs by category
 */
export function getPayoffsByCategory(category: PayoffCategory): PayoffInfo[] {
  return ALL_PAYOFFS.filter(p => p.category === category);
}

/**
 * Get payoffs by subcategory
 */
export function getPayoffsBySubcategory(subcategory: PayoffSubcategory): PayoffInfo[] {
  return ALL_PAYOFFS.filter(p => p.subcategory === subcategory);
}

/**
 * Get base payoffs only (no barriers)
 */
export function getBasePayoffs(): PayoffInfo[] {
  return ALL_PAYOFFS.filter(p => p.barrierType === 'None');
}

/**
 * Get barrier payoffs only
 */
export function getBarrierPayoffs(): PayoffInfo[] {
  return ALL_PAYOFFS.filter(p => p.barrierType !== 'None');
}

/**
 * Get payoff by name
 */
export function getPayoffByName(name: string): PayoffInfo | undefined {
  return ALL_PAYOFFS.find(p => p.name === name || p.abbreviation === name);
}

/**
 * Get all categories
 */
export function getAllCategories(): PayoffCategory[] {
  return ['Single Asset', 'Basket', 'Barrier Single Asset', 'Barrier Basket'];
}

/**
 * Get all subcategories
 */
export function getAllSubcategories(): PayoffSubcategory[] {
  return ['Simple', 'Asian', 'Lookback', 'Range', 'Dispersion', 'Rank'];
}

/**
 * Barrier type descriptions
 */
export const BARRIER_DESCRIPTIONS: Record<BarrierType, string> = {
  'None': 'No barrier',
  'UO': 'Up-and-Out: Option knocked out if price goes above barrier',
  'DO': 'Down-and-Out: Option knocked out if price goes below barrier',
  'UI': 'Up-and-In: Option activated if price goes above barrier',
  'DI': 'Down-and-In: Option activated if price goes below barrier',
  'UODO': 'Double Knock-Out: Knocked out if price exits corridor',
  'UIDI': 'Double Knock-In: Activated if price exits corridor',
  'UIDO': 'Up-In-Down-Out: Activated by upper barrier, knocked out by lower',
  'UODI': 'Up-Out-Down-In: Knocked out by upper barrier, activated by lower',
  'PTB': 'Partial Time Barrier: Barrier only active during specified time window',
  'StepB': 'Step Barrier: Time-varying barrier (grows at risk-free rate or random walk)',
  'DStepB': 'Double Step Barrier: Two time-varying barriers (corridor)'
};

/**
 * Summary statistics
 */
export const PAYOFF_STATS = {
  totalPayoffs: ALL_PAYOFFS.length,
  basePayoffs: BASE_PAYOFFS.length,
  barrierPayoffs: ALL_PAYOFFS.length - BASE_PAYOFFS.length,
  barrierTypes: BARRIER_TYPES.length - 1, // Exclude 'None'
  categories: getAllCategories().length,
  subcategories: getAllSubcategories().length
};
