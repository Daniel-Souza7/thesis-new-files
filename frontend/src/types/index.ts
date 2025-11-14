// Type definitions for Optimal Stopping Explorer

export interface PricingParams {
  algorithm: Algorithm;
  model: ModelType;
  payoff: PayoffType;
  nb_stocks: number;
  nb_paths: number;
  nb_dates: number;
  spot: number;
  strike: number;
  maturity: number;
  volatility: number;
  drift: number;
  rate: number;
  hidden_size?: number;
  nb_epochs?: number;
}

export interface PricingResult {
  price: number;
  time_path_gen: number;
  time_pricing: number;
  total_time: number;
}

export interface ComparisonResult {
  algorithm: string;
  price: number;
  time: number;
}

export interface PathData {
  paths: number[][][]; // [nb_paths, nb_stocks, nb_dates+1]
  time_grid: number[];
}

export type Algorithm = 'RLSM' | 'RFQI' | 'LSM' | 'NLSM' | 'DOS' | 'FQI';
export type ModelType = 'BlackScholes' | 'RealData';
export type PayoffType = 'MaxCall' | 'BasketCall' | 'MinCall' | 'GeometricBasketCall';

export interface AlgorithmInfo {
  name: string;
  fullName: string;
  description: string;
  formula: string;
  keyPoints: string[];
  reference: string;
}
