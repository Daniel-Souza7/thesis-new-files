import { motion } from 'framer-motion';
import { useState } from 'react';
import { Play, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';
import type { Algorithm, ModelType, PayoffType, PricingParams } from '../types';

interface PricingResult {
  price: number;
  time_path_gen: number;
  time_pricing: number;
  total_time: number;
}

const Demo = () => {
  // Algorithm and problem settings
  const [algorithm, setAlgorithm] = useState<Algorithm>('RLSM');
  const [payoff, setPayoff] = useState<PayoffType>('MaxCall');

  // Market parameters
  const [nbStocks, setNbStocks] = useState(5);
  const [spot, setSpot] = useState(100);
  const [strike, setStrike] = useState(100);
  const [volatility, setVolatility] = useState(0.2);
  const [drift, setDrift] = useState(0.05);
  const [rate, setRate] = useState(0.05);
  const [maturity, setMaturity] = useState(1.0);

  // Simulation parameters
  const [nbPaths, setNbPaths] = useState(10000);
  const [nbDates, setNbDates] = useState(10);

  // Neural network parameters
  const [hiddenSize, setHiddenSize] = useState(100);

  // Results
  const [result, setResult] = useState<PricingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runPricing = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const params: PricingParams = {
        algorithm,
        model: 'BlackScholes',
        payoff,
        nb_stocks: nbStocks,
        nb_paths: nbPaths,
        nb_dates: nbDates,
        spot,
        strike,
        maturity,
        volatility,
        drift,
        rate,
        hidden_size: hiddenSize,
        nb_epochs: 50,
      };

      const response = await axios.post('/api/price', params);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to price option');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="demo" className="section bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          {/* Section header */}
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Interactive Demo
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Experiment with parameters and see algorithms in action
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Controls */}
            <div className="glass rounded-2xl p-6 space-y-6">
              <h3 className="text-xl font-semibold text-gray-800">
                Parameters
              </h3>

              {/* Algorithm Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Algorithm
                </label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="RLSM">RLSM (Fast)</option>
                  <option value="RFQI">RFQI (Fast)</option>
                  <option value="LSM">LSM (Classic)</option>
                  <option value="NLSM">NLSM</option>
                  <option value="DOS">DOS (Deep)</option>
                  <option value="FQI">FQI</option>
                </select>
              </div>

              {/* Payoff Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Payoff Type
                </label>
                <select
                  value={payoff}
                  onChange={(e) => setPayoff(e.target.value as PayoffType)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="MaxCall">Max Call</option>
                  <option value="BasketCall">Basket Call</option>
                  <option value="MinCall">Min Call</option>
                  <option value="GeometricBasketCall">Geometric Basket Call</option>
                </select>
              </div>

              {/* Market Parameters */}
              <div className="pt-4 border-t border-gray-200">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">Market Parameters</h4>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Number of Stocks: {nbStocks}
                    </label>
                    <input
                      type="range" min="2" max="10" value={nbStocks}
                      onChange={(e) => setNbStocks(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Spot Price: ${spot}
                    </label>
                    <input
                      type="range" min="80" max="120" value={spot}
                      onChange={(e) => setSpot(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Strike Price: ${strike}
                    </label>
                    <input
                      type="range" min="80" max="120" value={strike}
                      onChange={(e) => setStrike(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Volatility: {(volatility * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range" min="0.1" max="0.5" step="0.05" value={volatility}
                      onChange={(e) => setVolatility(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Maturity: {maturity.toFixed(2)} years
                    </label>
                    <input
                      type="range" min="0.25" max="2" step="0.25" value={maturity}
                      onChange={(e) => setMaturity(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Simulation Parameters */}
              <div className="pt-4 border-t border-gray-200">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">Simulation</h4>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Paths: {nbPaths.toLocaleString()}
                    </label>
                    <input
                      type="range" min="1000" max="50000" step="1000" value={nbPaths}
                      onChange={(e) => setNbPaths(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Time Steps: {nbDates}
                    </label>
                    <input
                      type="range" min="5" max="20" value={nbDates}
                      onChange={(e) => setNbDates(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-600 mb-1">
                      Hidden Neurons: {hiddenSize}
                    </label>
                    <input
                      type="range" min="50" max="200" step="10" value={hiddenSize}
                      onChange={(e) => setHiddenSize(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              <button
                onClick={runPricing}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                <Play size={20} />
                {loading ? 'Running...' : 'Run Pricing'}
              </button>
            </div>

            {/* Results */}
            <div className="lg:col-span-2 glass rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-6 text-gray-800">
                Results
              </h3>

              {loading && (
                <div className="flex flex-col items-center justify-center h-96">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
                  <p className="text-gray-600">Pricing option...</p>
                </div>
              )}

              {error && (
                <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-900">Error</p>
                    <p className="text-sm text-red-700 mt-1">{error}</p>
                  </div>
                </div>
              )}

              {result && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="space-y-6"
                >
                  <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-green-900">Success</p>
                      <p className="text-sm text-green-700 mt-1">Option priced successfully</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-6 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl text-white">
                      <p className="text-sm opacity-90 mb-1">Option Price</p>
                      <p className="text-4xl font-bold">${result.price.toFixed(4)}</p>
                    </div>
                    <div className="p-6 bg-gradient-to-br from-green-500 to-green-600 rounded-xl text-white">
                      <p className="text-sm opacity-90 mb-1">Total Time</p>
                      <p className="text-4xl font-bold">{result.total_time.toFixed(3)}s</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Path Generation</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {result.time_path_gen.toFixed(3)}s
                      </p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-600 mb-1">Pricing Time</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {result.time_pricing.toFixed(3)}s
                      </p>
                    </div>
                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold text-gray-900 mb-2">Configuration</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div><span className="text-gray-600">Algorithm:</span> <span className="font-medium">{algorithm}</span></div>
                      <div><span className="text-gray-600">Payoff:</span> <span className="font-medium">{payoff}</span></div>
                      <div><span className="text-gray-600">Spot:</span> <span className="font-medium">${spot}</span></div>
                      <div><span className="text-gray-600">Strike:</span> <span className="font-medium">${strike}</span></div>
                      <div><span className="text-gray-600">Volatility:</span> <span className="font-medium">{(volatility * 100).toFixed(0)}%</span></div>
                      <div><span className="text-gray-600">Maturity:</span> <span className="font-medium">{maturity}y</span></div>
                      <div><span className="text-gray-600">Paths:</span> <span className="font-medium">{nbPaths.toLocaleString()}</span></div>
                      <div><span className="text-gray-600">Time Steps:</span> <span className="font-medium">{nbDates}</span></div>
                    </div>
                  </div>
                </motion.div>
              )}

              {!loading && !error && !result && (
                <div className="flex flex-col items-center justify-center h-96 text-center">
                  <Play size={64} className="text-gray-300 mb-4" />
                  <p className="text-gray-500 text-lg">
                    Configure parameters and click "Run Pricing" to price an American option
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Demo;
