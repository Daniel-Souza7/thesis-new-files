/**
 * Pricing API endpoint for option valuation.
 *
 * Supports multiple models including RealData with block bootstrap.
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

const PYTHON_PATH = '/usr/local/bin/python3';
const PRICING_ENGINE_SCRIPT = path.join(process.cwd(), 'api', 'pricing_engine.py');

/**
 * Execute Python pricing engine.
 */
async function executePricingEngine(
  command: string,
  params?: any,
  onProgress?: (data: string) => void
): Promise<any> {
  return new Promise((resolve, reject) => {
    const args = [PRICING_ENGINE_SCRIPT, command];
    if (params) {
      args.push(JSON.stringify(params));
    }

    const pythonProcess = spawn(PYTHON_PATH, args);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      stdout += chunk;

      // Call progress callback if provided
      if (onProgress) {
        onProgress(chunk);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderr += chunk;

      // Also send stderr to progress (for status messages)
      if (onProgress) {
        onProgress(chunk);
      }
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse JSON output: ${stdout}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * POST /api/price
 * Price an option using specified model and parameters.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    const requiredFields = ['model_type', 'payoff_type', 'algorithm'];
    for (const field of requiredFields) {
      if (!body[field]) {
        return NextResponse.json(
          {
            success: false,
            error: `Missing required field: ${field}`,
          },
          { status: 400 }
        );
      }
    }

    // Price the option
    const result = await executePricingEngine('price', body);

    return NextResponse.json(result);
  } catch (error) {
    console.error('Error pricing option:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/price/stream
 * Price an option with streaming progress updates.
 */
export async function OPTIONS(request: NextRequest) {
  // Handle CORS preflight
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

/**
 * GET /api/price/models
 * Get available models and their parameters.
 */
export async function GET(request: NextRequest) {
  const models = {
    BlackScholes: {
      name: 'Black-Scholes',
      description: 'Geometric Brownian motion with constant drift and volatility',
      parameters: {
        required: ['spot', 'drift', 'volatility', 'rate', 'maturity'],
        optional: ['nb_stocks', 'nb_dates', 'nb_paths'],
      },
    },
    Heston: {
      name: 'Heston',
      description: 'Stochastic volatility model',
      parameters: {
        required: ['spot', 'drift', 'rate', 'maturity', 'kappa', 'theta', 'xi', 'rho', 'v0'],
        optional: ['nb_stocks', 'nb_dates', 'nb_paths'],
      },
    },
    FractionalBlackScholes: {
      name: 'Fractional Black-Scholes',
      description: 'Fractional Brownian motion with long memory',
      parameters: {
        required: ['spot', 'drift', 'volatility', 'rate', 'maturity', 'hurst'],
        optional: ['nb_stocks', 'nb_dates', 'nb_paths'],
      },
    },
    RoughHeston: {
      name: 'Rough Heston',
      description: 'Rough volatility model with Hurst < 0.5',
      parameters: {
        required: ['spot', 'drift', 'rate', 'maturity', 'hurst', 'kappa', 'theta', 'xi', 'rho', 'v0'],
        optional: ['nb_stocks', 'nb_dates', 'nb_paths'],
      },
    },
    RealData: {
      name: 'Real Data (Block Bootstrap)',
      description: 'Real market data with stationary block bootstrap preserving autocorrelation',
      parameters: {
        required: ['tickers', 'spot', 'rate', 'maturity'],
        optional: [
          'start_date',
          'end_date',
          'drift_override',
          'volatility_override',
          'exclude_crisis',
          'only_crisis',
          'nb_stocks',
          'nb_dates',
          'nb_paths',
        ],
      },
      defaults: {
        start_date: '2010-01-01',
        end_date: '2024-01-01',
        drift_override: null, // Use empirical
        volatility_override: null, // Use empirical
        exclude_crisis: false,
        only_crisis: false,
      },
      notes: [
        'Set drift_override=null and volatility_override=null to use empirical values',
        'Set drift_override and volatility_override to specific values to override',
        'Tickers must be valid stock symbols (e.g., AAPL, MSFT)',
        'Date range determines available historical data',
        'Block bootstrap preserves autocorrelation and volatility clustering',
      ],
    },
  };

  const algorithms = {
    standard: {
      RLSM: 'Randomized Least Squares Monte Carlo',
      RFQI: 'Randomized Fitted Q-Iteration',
      LSM: 'Least Squares Monte Carlo',
      FQI: 'Fitted Q-Iteration',
    },
    path_dependent: {
      SRLSM: 'State-augmented RLSM (for path-dependent options)',
      SRFQI: 'State-augmented RFQI (for path-dependent options)',
    },
  };

  return NextResponse.json({
    success: true,
    models,
    algorithms,
  });
}
