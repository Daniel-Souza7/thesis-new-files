'use client';

import React, { useState } from 'react';
import { ArcadeCabinet, RetroButton, RetroPanel } from '@/components/ui';
import { PayoffSelector } from '@/components/ui/PayoffSelector';
import type { PayoffInfo } from '@/lib/payoffs';

interface PricingResult {
  price: number;
  computation_time: number;
  exercise_time?: number;
  model_params?: Record<string, any>;
  error?: string;
}

export default function CalculatorPage() {
  // Model and Algorithm Selection
  const [model, setModel] = useState('BlackScholes');
  const [algorithm, setAlgorithm] = useState('RLSM');

  // Payoff Selection
  const [selectedPayoff, setSelectedPayoff] = useState<PayoffInfo | null>(null);
  const [payoffParameters, setPayoffParameters] = useState<Record<string, any>>({});

  // Market Parameters
  const [spotPrice, setSpotPrice] = useState(100);
  const [strikePrice, setStrikePrice] = useState(100);
  const [volatility, setVolatility] = useState(0.2);
  const [drift, setDrift] = useState(0.06);
  const [riskFreeRate, setRiskFreeRate] = useState(0.02);
  const [maturity, setMaturity] = useState(1.0);

  // Computational Parameters
  const [nbPaths, setNbPaths] = useState(10000);
  const [nbDates, setNbDates] = useState(10);
  const [hiddenSize, setHiddenSize] = useState(20);
  const [epochs, setEpochs] = useState(30);

  // UI State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PricingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePayoffSelect = (payoff: PayoffInfo, parameters: Record<string, any>) => {
    setSelectedPayoff(payoff);
    setPayoffParameters(parameters);

    // Auto-switch algorithm based on path dependency
    if (payoff.isPathDependent) {
      // Switch to path-dependent algorithm if current is not
      if (!['SRLSM', 'SRFQI'].includes(algorithm)) {
        setAlgorithm('SRLSM');
      }
    } else {
      // Switch to standard algorithm if current is path-dependent
      if (['SRLSM', 'SRFQI'].includes(algorithm)) {
        setAlgorithm('RLSM');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedPayoff) {
      setError('Please select a payoff');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestBody = {
        model_type: model,
        algorithm,
        payoff_type: selectedPayoff.name,
        spot_price: spotPrice,
        strike: payoffParameters.strike || strikePrice,
        volatility,
        drift,
        rate: riskFreeRate,
        maturity,
        nb_paths: nbPaths,
        nb_dates: nbDates,
        nb_stocks: selectedPayoff.requiresMultipleAssets ? 5 : 1,
        hidden_size: hiddenSize,
        epochs,
        ...payoffParameters,
      };

      const response = await fetch('/api/price', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to compute option price');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Shared input styles
  const inputStyles: React.CSSProperties = {
    width: '100%',
    padding: '0.75rem',
    background: 'rgba(0, 0, 0, 0.6)',
    border: '2px solid #00ff00',
    borderRadius: '4px',
    color: '#00ff00',
    fontFamily: "'Courier New', monospace",
    fontSize: '0.9rem',
    boxShadow: '0 0 10px rgba(0, 255, 0, 0.3), inset 0 0 10px rgba(0, 0, 0, 0.5)',
  };

  const labelStyles: React.CSSProperties = {
    display: 'block',
    marginBottom: '0.5rem',
    fontFamily: "'Press Start 2P', cursive",
    fontSize: '0.6rem',
    color: '#00ffff',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    textShadow: '0 0 5px #00ffff',
  };

  return (
    <ArcadeCabinet title="OPTION PRICING CALCULATOR">
      <form onSubmit={handleSubmit} style={{ padding: '1rem' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem',
        }}>
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Model Selection */}
            <RetroPanel title="Model Selection" borderColor="green">
              <label style={labelStyles}>Stock Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                style={inputStyles}
              >
                <option value="BlackScholes">Black-Scholes</option>
                <option value="Heston">Heston</option>
                <option value="RoughHeston">Rough Heston</option>
                <option value="RealData">Real Data</option>
                <option value="FractionalBlackScholes">Fractional Black-Scholes</option>
              </select>
            </RetroPanel>

            {/* Algorithm Selection */}
            <RetroPanel title="Algorithm Selection" borderColor="cyan">
              <label style={labelStyles}>Pricing Algorithm</label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                style={inputStyles}
              >
                {selectedPayoff?.isPathDependent ? (
                  // Path-dependent algorithms
                  <>
                    <option value="SRLSM">SRLSM (Path-Dependent)</option>
                    <option value="SRFQI">SRFQI (Path-Dependent)</option>
                    <option value="LSM">LSM</option>
                    <option value="FQI">FQI</option>
                    <option value="EOP">EOP</option>
                  </>
                ) : (
                  // Standard algorithms
                  <>
                    <option value="RLSM">RLSM</option>
                    <option value="RFQI">RFQI</option>
                    <option value="LSM">LSM</option>
                    <option value="FQI">FQI</option>
                    <option value="EOP">EOP</option>
                  </>
                )}
              </select>
            </RetroPanel>

            {/* Market Parameters */}
            <RetroPanel title="Market Parameters" borderColor="yellow">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={labelStyles}>Spot Price</label>
                  <input
                    type="number"
                    step="0.01"
                    value={spotPrice}
                    onChange={(e) => setSpotPrice(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Strike Price</label>
                  <input
                    type="number"
                    step="0.01"
                    value={strikePrice}
                    onChange={(e) => setStrikePrice(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Volatility</label>
                  <input
                    type="number"
                    step="0.01"
                    value={volatility}
                    onChange={(e) => setVolatility(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Drift</label>
                  <input
                    type="number"
                    step="0.01"
                    value={drift}
                    onChange={(e) => setDrift(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Risk-Free Rate</label>
                  <input
                    type="number"
                    step="0.01"
                    value={riskFreeRate}
                    onChange={(e) => setRiskFreeRate(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Maturity (Years)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={maturity}
                    onChange={(e) => setMaturity(parseFloat(e.target.value))}
                    style={inputStyles}
                  />
                </div>
              </div>
            </RetroPanel>

            {/* Computational Parameters */}
            <RetroPanel title="Computational Parameters" borderColor="orange">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label style={labelStyles}>Number of Paths</label>
                  <input
                    type="number"
                    step="1000"
                    value={nbPaths}
                    onChange={(e) => setNbPaths(parseInt(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Number of Dates</label>
                  <input
                    type="number"
                    step="1"
                    value={nbDates}
                    onChange={(e) => setNbDates(parseInt(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Hidden Size</label>
                  <input
                    type="number"
                    step="5"
                    value={hiddenSize}
                    onChange={(e) => setHiddenSize(parseInt(e.target.value))}
                    style={inputStyles}
                  />
                </div>

                <div>
                  <label style={labelStyles}>Epochs</label>
                  <input
                    type="number"
                    step="10"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    style={inputStyles}
                  />
                </div>
              </div>
            </RetroPanel>
          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Payoff Selection */}
            <PayoffSelector onPayoffSelect={handlePayoffSelect} />

            {/* Results Panel */}
            {(result || error) && (
              <RetroPanel
                title={error ? 'ERROR' : 'RESULTS'}
                borderColor={error ? 'orange' : 'green'}
              >
                {error ? (
                  <div style={{
                    fontFamily: "'Courier New', monospace",
                    color: '#ff6600',
                    fontSize: '0.9rem',
                    lineHeight: '1.6',
                  }}>
                    <p style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>ERROR:</p>
                    <p>{error}</p>
                  </div>
                ) : result ? (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{
                      fontFamily: "'Press Start 2P', cursive",
                      fontSize: '0.7rem',
                      color: '#00ffff',
                      marginBottom: '1rem',
                      textTransform: 'uppercase',
                    }}>
                      Option Price
                    </div>
                    <div style={{
                      fontFamily: "'Press Start 2P', cursive",
                      fontSize: '2.5rem',
                      color: '#00ff00',
                      textShadow: '0 0 20px #00ff00, 0 0 40px #00ff00',
                      marginBottom: '1.5rem',
                    }}>
                      ${result.price.toFixed(4)}
                    </div>

                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '1rem',
                      borderTop: '2px solid #00ffff',
                      paddingTop: '1rem',
                      marginTop: '1rem',
                    }}>
                      <div>
                        <p style={{
                          fontFamily: "'Press Start 2P', cursive",
                          fontSize: '0.5rem',
                          color: '#00ffff',
                          marginBottom: '0.5rem',
                        }}>
                          COMP TIME
                        </p>
                        <p style={{
                          fontFamily: "'Courier New', monospace",
                          fontSize: '1rem',
                          color: '#00ff00',
                        }}>
                          {result.computation_time.toFixed(3)}s
                        </p>
                      </div>

                      {result.exercise_time !== undefined && (
                        <div>
                          <p style={{
                            fontFamily: "'Press Start 2P', cursive",
                            fontSize: '0.5rem',
                            color: '#00ffff',
                            marginBottom: '0.5rem',
                          }}>
                            EXERCISE TIME
                          </p>
                          <p style={{
                            fontFamily: "'Courier New', monospace",
                            fontSize: '1rem',
                            color: '#00ff00',
                          }}>
                            {result.exercise_time.toFixed(3)}
                          </p>
                        </div>
                      )}
                    </div>

                    {result.model_params && (
                      <div style={{
                        borderTop: '2px solid #00ffff',
                        paddingTop: '1rem',
                        marginTop: '1rem',
                      }}>
                        <p style={{
                          fontFamily: "'Press Start 2P', cursive",
                          fontSize: '0.5rem',
                          color: '#00ffff',
                          marginBottom: '0.75rem',
                        }}>
                          MODEL PARAMETERS
                        </p>
                        <div style={{
                          fontFamily: "'Courier New', monospace",
                          fontSize: '0.8rem',
                          color: '#00ff00',
                          textAlign: 'left',
                        }}>
                          {Object.entries(result.model_params).map(([key, value]) => (
                            <div key={key} style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              marginBottom: '0.25rem',
                            }}>
                              <span style={{ color: '#00ffff' }}>{key}:</span>
                              <span>{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </RetroPanel>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          {loading ? (
            <div style={{
              fontFamily: "'Press Start 2P', cursive",
              fontSize: '1.5rem',
              color: '#00ff00',
              textShadow: '0 0 20px #00ff00',
              animation: 'pulse 1s infinite',
            }}>
              COMPUTING...
            </div>
          ) : (
            <RetroButton
              type="submit"
              variant="green"
              size="lg"
              disabled={!selectedPayoff}
            >
              PRICE OPTION
            </RetroButton>
          )}
        </div>
      </form>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </ArcadeCabinet>
  );
}
