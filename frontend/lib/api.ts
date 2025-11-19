/**
 * Frontend API client for option pricing
 *
 * Provides TypeScript interfaces and functions to interact with the pricing API.
 */

// ==================== TypeScript Interfaces ====================

export interface PricingRequest {
  algorithm: string;
  model_type: string;
  payoff_type: string;
  // Model parameters
  spot?: number;
  drift?: number;
  volatility?: number;
  rate?: number;
  nb_stocks?: number;
  nb_paths?: number;
  nb_dates?: number;
  maturity?: number;
  dividend?: number;
  // Payoff parameters
  strike?: number;
  barrier?: number;
  barriers_up?: number;
  barriers_down?: number;
  k?: number;
  weights?: number[];
  step_param1?: number;
  step_param2?: number;
  step_param3?: number;
  step_param4?: number;
  // Algorithm parameters
  hidden_size?: number;
  nb_epochs?: number;
  factors?: number[];
  train_ITM_only?: boolean;
  use_payoff_as_input?: boolean;
  // RealData-specific parameters
  tickers?: string[];
  start_date?: string;
  end_date?: string;
  drift_override?: number | null;
  volatility_override?: number | null;
  exclude_crisis?: boolean;
  only_crisis?: boolean;
  // Heston parameters
  kappa?: number;
  theta?: number;
  xi?: number;
  rho?: number;
  v0?: number;
  // Other model parameters
  hurst?: number;
}

export interface ModelInfo {
  type: string;
  spot: number;
  drift: number | null;
  volatility: number | null;
  rate: number;
  nb_stocks: number;
  maturity: number;
  nb_dates: number;
  nb_paths: number;
  tickers?: string[];
  empirical_drift?: number;
  empirical_volatility?: number;
  block_length?: number;
  data_days?: number;
  drift_override?: number;
  volatility_override?: number;
}

export interface PayoffInfo {
  type: string;
  strike: number;
  is_path_dependent: boolean;
}

export interface PricingResponse {
  success: boolean;
  price?: number;
  computation_time?: number;
  exercise_time?: number | null;
  paths_sample?: number[][][]; // Array of paths, each path is [[time, price], ...]
  model_info?: ModelInfo;
  payoff_info?: PayoffInfo;
  algorithm?: string;
  error?: string;
  error_type?: string;
}

export interface PayoffListResponse {
  success: boolean;
  payoffs?: string[];
  error?: string;
}

export interface PayoffDetailsResponse {
  success: boolean;
  name?: string;
  abbreviation?: string;
  is_path_dependent?: boolean;
  required_params?: string[];
  optional_params?: string[];
  error?: string;
}

export interface StockInfoRequest {
  tickers: string[];
  start_date?: string;
  end_date?: string;
}

export interface StockStatistics {
  ticker: string;
  empirical_drift_annual: number;
  empirical_volatility_annual: number;
}

export interface StockInfoResponse {
  success: boolean;
  tickers?: string[];
  start_date?: string;
  end_date?: string;
  data_days?: number;
  block_length?: number;
  overall_drift?: number;
  overall_volatility?: number;
  stock_statistics?: StockStatistics[];
  correlation_matrix?: number[][];
  error?: string;
  error_type?: string;
}

// ==================== API Functions ====================

/**
 * Price an option using the specified algorithm, model, and payoff
 */
export async function priceOption(params: PricingRequest): Promise<PricingResponse> {
  try {
    const response = await fetch('/api/price', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.error || `HTTP ${response.status}: ${response.statusText}`,
        error_type: errorData.error_type || 'HTTPError',
      };
    }

    const data: PricingResponse = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      error_type: 'NetworkError',
    };
  }
}

/**
 * Get list of all available payoffs
 */
export async function getPayoffs(): Promise<PayoffListResponse> {
  try {
    const response = await fetch('/api/payoffs');

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.error || `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data: PayoffListResponse = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

/**
 * Get information about a specific payoff
 */
export async function getPayoffDetails(payoffName: string): Promise<PayoffDetailsResponse> {
  try {
    const encodedName = encodeURIComponent(payoffName);
    const response = await fetch(`/api/payoffs?name=${encodedName}`);

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.error || `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data: PayoffDetailsResponse = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

/**
 * Get information about stocks including empirical drift/volatility
 */
export async function getStockInfo(params: StockInfoRequest): Promise<StockInfoResponse> {
  try {
    const response = await fetch('/api/stock-info', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        success: false,
        error: errorData.error || `HTTP ${response.status}: ${response.statusText}`,
        error_type: errorData.error_type || 'HTTPError',
      };
    }

    const data: StockInfoResponse = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      error_type: 'NetworkError',
    };
  }
}

// ==================== Validation Helpers ====================

/**
 * Validate pricing request parameters
 */
export function validatePricingRequest(params: PricingRequest): string | null {
  // Check required fields
  if (!params.algorithm) {
    return 'algorithm is required';
  }

  if (!params.model_type) {
    return 'model_type is required';
  }

  if (!params.payoff_type) {
    return 'payoff_type is required';
  }

  // Validate algorithm
  const validAlgorithms = ['RLSM', 'RFQI', 'SRLSM', 'SRFQI', 'LSM', 'FQI', 'EOP'];
  if (!validAlgorithms.includes(params.algorithm)) {
    return `algorithm must be one of: ${validAlgorithms.join(', ')}`;
  }

  // Validate model
  const validModels = ['BlackScholes', 'Heston', 'FractionalBlackScholes', 'RoughHeston', 'RealData'];
  if (!validModels.includes(params.model_type)) {
    return `model_type must be one of: ${validModels.join(', ')}`;
  }

  // Validate numeric parameters
  if (params.spot !== undefined && params.spot <= 0) {
    return 'spot must be positive';
  }

  if (params.strike !== undefined && params.strike <= 0) {
    return 'strike must be positive';
  }

  if (params.nb_paths !== undefined && params.nb_paths <= 0) {
    return 'nb_paths must be positive';
  }

  if (params.nb_dates !== undefined && params.nb_dates <= 0) {
    return 'nb_dates must be positive';
  }

  if (params.maturity !== undefined && params.maturity <= 0) {
    return 'maturity must be positive';
  }

  // Model-specific validation
  if (params.model_type === 'RealData') {
    if (!params.tickers || params.tickers.length === 0) {
      return 'tickers is required for RealData model';
    }
  }

  return null; // Valid
}

// ==================== Default Values ====================

export const DEFAULT_PRICING_PARAMS: Partial<PricingRequest> = {
  spot: 100,
  strike: 100,
  drift: 0.05,
  volatility: 0.2,
  rate: 0.03,
  nb_stocks: 1,
  nb_paths: 10000,
  nb_dates: 50,
  maturity: 1.0,
  dividend: 0,
  hidden_size: 100,
  nb_epochs: 20,
  factors: [1.0, 1.0],
  train_ITM_only: true,
  use_payoff_as_input: false,
};

export const ALGORITHM_INFO = {
  RLSM: {
    name: 'Randomized LSM',
    description: 'Randomized Least Squares Monte Carlo for standard options',
    path_dependent: false,
  },
  RFQI: {
    name: 'Randomized FQI',
    description: 'Randomized Fitted Q-Iteration for standard options',
    path_dependent: false,
  },
  SRLSM: {
    name: 'State-augmented RLSM',
    description: 'RLSM with full path history for path-dependent options',
    path_dependent: true,
  },
  SRFQI: {
    name: 'State-augmented RFQI',
    description: 'RFQI with full path history for path-dependent options',
    path_dependent: true,
  },
  LSM: {
    name: 'Least Squares Monte Carlo',
    description: 'Classic LSM algorithm',
    path_dependent: false,
  },
  FQI: {
    name: 'Fitted Q-Iteration',
    description: 'Classic FQI algorithm',
    path_dependent: false,
  },
  EOP: {
    name: 'European Option',
    description: 'European option pricer (exercise only at maturity)',
    path_dependent: false,
  },
};

export const MODEL_INFO = {
  BlackScholes: {
    name: 'Black-Scholes',
    description: 'Geometric Brownian motion with constant drift and volatility',
  },
  Heston: {
    name: 'Heston',
    description: 'Stochastic volatility model',
  },
  FractionalBlackScholes: {
    name: 'Fractional Black-Scholes',
    description: 'Fractional Brownian motion with long memory',
  },
  RoughHeston: {
    name: 'Rough Heston',
    description: 'Rough volatility model with Hurst < 0.5',
  },
  RealData: {
    name: 'Real Data (Block Bootstrap)',
    description: 'Real market data with stationary block bootstrap',
  },
};
