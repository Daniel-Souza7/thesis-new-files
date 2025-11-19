/**
 * Next.js API Route: /api/payoffs
 *
 * Returns information about available payoffs and their parameters.
 */

import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

const PYTHON_PATH = 'python3';
const PRICING_ENGINE_SCRIPT = '/home/user/thesis-new-files/frontend/api/pricing_engine.py';

/**
 * Call Python pricing engine to get payoff information
 */
async function callPythonEngine(command: string, params?: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const args = [PRICING_ENGINE_SCRIPT, command];

    if (params) {
      args.push(JSON.stringify(params));
    }

    const pythonProcess = spawn(PYTHON_PATH, args, { shell: true });

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
        console.error('Python stderr:', stderr);
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch (error) {
        console.error('Failed to parse Python output:', stdout);
        reject(new Error(`Failed to parse Python output: ${error}`));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });

    // Set timeout
    const timeout = setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Python process timed out'));
    }, 30000);

    pythonProcess.on('close', () => {
      clearTimeout(timeout);
    });
  });
}

/**
 * GET /api/payoffs
 *
 * Returns list of all available payoffs or info about a specific payoff.
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const payoffName = searchParams.get('name');

    if (payoffName) {
      // Get info about specific payoff
      const result = await callPythonEngine('payoff_info', { payoff_name: payoffName });

      if (result.success) {
        return NextResponse.json(result, { status: 200 });
      } else {
        return NextResponse.json(result, { status: 404 });
      }
    } else {
      // Get list of all payoffs
      const result = await callPythonEngine('list_payoffs');

      if (result.success) {
        return NextResponse.json(result, { status: 200 });
      } else {
        return NextResponse.json(result, { status: 500 });
      }
    }
  } catch (error) {
    console.error('Error in /api/payoffs:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      },
      { status: 500 }
    );
  }
}
