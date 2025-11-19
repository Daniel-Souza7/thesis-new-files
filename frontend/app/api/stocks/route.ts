/**
 * Stock API endpoints for managing ticker data.
 *
 * Endpoints:
 * - GET /api/stocks - Get pre-loaded tickers
 * - POST /api/stocks/info - Get detailed ticker information
 * - POST /api/stocks/validate - Validate ticker symbols
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

const PYTHON_PATH = 'python3';
const STOCK_DATA_SCRIPT = path.join(process.cwd(), '..', 'api', 'stock_data.py');

/**
 * Execute Python stock data script.
 */
async function executePythonScript(
  command: string,
  params?: any
): Promise<any> {
  return new Promise((resolve, reject) => {
    const args = [STOCK_DATA_SCRIPT, command];
    if (params) {
      args.push(JSON.stringify(params));
    }

    const pythonProcess = spawn(PYTHON_PATH, args);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
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
 * GET /api/stocks
 * Get pre-loaded ticker symbols.
 */
export async function GET(request: NextRequest) {
  try {
    const result = await executePythonScript('preloaded');
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error fetching pre-loaded tickers:', error);
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
 * POST /api/stocks
 * Get detailed information about specific tickers or validate tickers.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, tickers, start_date, end_date } = body;

    if (!action) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required field: action (info, validate)',
        },
        { status: 400 }
      );
    }

    // Determine command based on action
    let command: string;
    if (action === 'info') {
      command = 'info';
    } else if (action === 'validate') {
      command = 'validate';
    } else {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid action: ${action}. Use 'info' or 'validate'`,
        },
        { status: 400 }
      );
    }

    // Build parameters
    const params: any = {};
    if (tickers) {
      params.tickers = tickers;
    }
    if (start_date) {
      params.start_date = start_date;
    }
    if (end_date) {
      params.end_date = end_date;
    }

    // Execute command
    const result = await executePythonScript(command, params);

    return NextResponse.json(result);
  } catch (error) {
    console.error('Error processing stock request:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
