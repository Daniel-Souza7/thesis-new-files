import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';

interface PathPoint {
  time: number;
  price: number;
}

const DoubleBarrierVisualization = () => {
  const [selectedPath, setSelectedPath] = useState<number>(0);
  const [paths, setPaths] = useState<PathPoint[][]>([]);
  const [pathStatus, setPathStatus] = useState<string[]>([]);

  const upperBarrier = 115;
  const lowerBarrier = 85;
  const K = 100;

  useEffect(() => {
    generatePaths();
  }, []);

  const generatePaths = () => {
    const numPaths = 5;
    const numSteps = 50;
    const S0 = 100;
    const sigma = 0.25;
    const dt = 1 / (numSteps - 1);

    const generatedPaths: PathPoint[][] = [];
    const statuses: string[] = [];

    for (let p = 0; p < numPaths; p++) {
      const path: PathPoint[] = [];
      let S = S0;
      let hitBarrier = false;

      for (let i = 0; i < numSteps; i++) {
        path.push({ time: i * dt, price: S });

        // Check barrier breach
        if (!hitBarrier && (S >= upperBarrier || S <= lowerBarrier)) {
          hitBarrier = true;
        }

        if (i < numSteps - 1) {
          const z = (Math.random() - 0.5) * 3;
          S = S * Math.exp(sigma * Math.sqrt(dt) * z);
        }
      }

      generatedPaths.push(path);

      if (hitBarrier) {
        statuses.push('Knocked Out (Worthless)');
      } else {
        const finalPrice = path[path.length - 1].price;
        const payoff = Math.max(finalPrice - K, 0);
        statuses.push(payoff > 0 ? `Active (Payoff: $${payoff.toFixed(2)})` : 'Active (Out of money)');
      }
    }

    setPaths(generatedPaths);
    setPathStatus(statuses);
  };

  const currentPath = paths[selectedPath] || [];
  const isKnockedOut = pathStatus[selectedPath]?.includes('Knocked Out');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-2">Double Barrier Knock-Out Call</h3>
      <p className="text-gray-600 mb-4">
        Path-dependent option that becomes worthless if the price touches either barrier
      </p>

      {/* Option specification */}
      <div className="mb-4 p-3 bg-gradient-to-r from-orange-50 to-red-50 rounded-lg border border-orange-200">
        <p className="text-sm text-gray-700">
          <strong>Option:</strong> Double Barrier Knock-Out Call | <strong>Strike:</strong> K=$100 |
          <strong> Upper Barrier:</strong> $115 | <strong>Lower Barrier:</strong> $85 |
          <strong> Initial Price:</strong> S₀=$100
        </p>
        <p className="text-xs text-gray-600 mt-1">
          If the stock price touches either barrier at any time, the option becomes worthless immediately.
        </p>
      </div>

      <div className="flex gap-4 mb-6">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Path
          </label>
          <select
            value={selectedPath}
            onChange={(e) => setSelectedPath(Number(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {paths.map((_, idx) => (
              <option key={idx} value={idx}>
                Path {idx + 1} - {pathStatus[idx]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {currentPath.length > 0 && (
        <>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={currentPath} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="time"
                type="number"
                domain={[0, 1]}
                label={{ value: 'Time (years)', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                domain={[75, 125]}
                label={{ value: 'Stock Price ($)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
                labelFormatter={(label) => `Time: ${label.toFixed(2)}y`}
              />
              <Legend />

              {/* Barrier zones */}
              <Area
                type="monotone"
                dataKey={() => 125}
                fill="#fca5a5"
                fillOpacity={0.2}
                stroke="none"
                name="Knock-Out Zone (Upper)"
              />
              <Area
                type="monotone"
                dataKey={() => 75}
                fill="#fca5a5"
                fillOpacity={0.2}
                stroke="none"
                name="Knock-Out Zone (Lower)"
              />

              {/* Barriers */}
              <ReferenceLine
                y={upperBarrier}
                stroke="#dc2626"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Upper Barrier ($115)', position: 'right', fill: '#dc2626' }}
              />
              <ReferenceLine
                y={lowerBarrier}
                stroke="#dc2626"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Lower Barrier ($85)', position: 'right', fill: '#dc2626' }}
              />

              {/* Strike */}
              <ReferenceLine
                y={K}
                stroke="#64748b"
                strokeDasharray="3 3"
                label={{ value: 'Strike ($100)', position: 'left' }}
              />

              {/* Stock price path */}
              <Line
                type="monotone"
                dataKey="price"
                stroke={isKnockedOut ? '#ef4444' : '#3b82f6'}
                strokeWidth={3}
                dot={false}
                name="Stock Price Path"
              />
            </ComposedChart>
          </ResponsiveContainer>

          {/* Path status */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`p-4 rounded-lg ${isKnockedOut ? 'bg-red-50' : 'bg-green-50'}`}>
              <p className="text-sm text-gray-600 mb-1">Path Status</p>
              <p className={`text-xl font-bold ${isKnockedOut ? 'text-red-600' : 'text-green-600'}`}>
                {isKnockedOut ? 'Knocked Out' : 'Active'}
              </p>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Final Stock Price</p>
              <p className="text-xl font-bold text-blue-600">
                ${currentPath[currentPath.length - 1]?.price.toFixed(2)}
              </p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Option Payoff</p>
              <p className="text-xl font-bold text-purple-600">
                {isKnockedOut
                  ? '$0.00 (Barrier Hit)'
                  : `$${Math.max(currentPath[currentPath.length - 1]?.price - K, 0).toFixed(2)}`}
              </p>
            </div>
          </div>
        </>
      )}

      <div className="mt-4 p-4 bg-orange-50 rounded-lg">
        <p className="text-sm text-gray-700">
          <strong>Why This Is Harder:</strong> The payoff depends on the entire price history, not just
          the final price. Standard algorithms assume Markovian dynamics (future depends only on present),
          but barrier options require tracking whether a barrier was ever hit. This is where{' '}
          <strong>SRLSM</strong> and <strong>SRFQI</strong> excel - they use recurrent neural networks
          to remember the path history.
        </p>
      </div>
    </motion.div>
  );
};

export default DoubleBarrierVisualization;
