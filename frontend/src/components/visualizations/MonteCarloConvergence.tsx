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
} from 'recharts';

interface ConvergencePoint {
  paths: number;
  price: number;
  std: number;
}

const MonteCarloConvergence = () => {
  const [data, setData] = useState<ConvergencePoint[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const truePrice = 11.26; // Reference price

  useEffect(() => {
    generateConvergenceData();
  }, []);

  const generateConvergenceData = () => {
    const pathCounts = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    const convergenceData: ConvergencePoint[] = pathCounts.map((n) => {
      // Simulate convergence: price approaches true value as n increases
      const error = 2.0 / Math.sqrt(n);
      const price = truePrice + (Math.random() - 0.5) * error;
      const std = error / 2;
      return { paths: n, price, std };
    });
    setData(convergenceData);
  };

  const animate = () => {
    setIsAnimating(true);
    setData([]);

    const pathCounts = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
    let index = 0;

    const interval = setInterval(() => {
      if (index >= pathCounts.length) {
        clearInterval(interval);
        setIsAnimating(false);
        return;
      }

      const n = pathCounts[index];
      const error = 2.0 / Math.sqrt(n);
      const price = truePrice + (Math.random() - 0.5) * error;
      const std = error / 2;

      setData((prev) => [...prev, { paths: n, price, std }]);
      index++;
    }, 300);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg p-6"
    >
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-xl font-semibold">Monte Carlo Convergence</h3>
          <p className="text-gray-600 mt-1">
            Price estimate stabilizes as we increase the number of simulation paths
          </p>
        </div>
        <button
          onClick={animate}
          disabled={isAnimating}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isAnimating ? 'Animating...' : 'Replay Animation'}
        </button>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="paths"
            scale="log"
            domain={[100, 100000]}
            tickFormatter={(value) => {
              if (value >= 1000) return `${value / 1000}k`;
              return value.toString();
            }}
            label={{ value: 'Number of Paths', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            domain={[10.5, 12.0]}
            label={{ value: 'Estimated Price ($)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            formatter={(value: number) => [`$${value.toFixed(4)}`, 'Price']}
            labelFormatter={(label) => `Paths: ${label.toLocaleString()}`}
          />
          <Legend />
          <ReferenceLine
            y={truePrice}
            stroke="#ef4444"
            strokeDasharray="5 5"
            label={{ value: 'True Price', position: 'right' }}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#3b82f6"
            strokeWidth={3}
            dot={{ r: 5 }}
            name="Estimated Price"
            isAnimationActive={true}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Initial Estimate (100 paths)</p>
          <p className="text-2xl font-bold text-blue-600">
            ${data[0]?.price.toFixed(4) || '---'}
          </p>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Final Estimate (100k paths)</p>
          <p className="text-2xl font-bold text-green-600">
            ${data[data.length - 1]?.price.toFixed(4) || '---'}
          </p>
        </div>
        <div className="p-4 bg-red-50 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">True Price (Reference)</p>
          <p className="text-2xl font-bold text-red-600">${truePrice.toFixed(4)}</p>
        </div>
      </div>

      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <p className="text-sm text-gray-700">
          <strong>Key Insight:</strong> The standard error decreases as 1/√n, so doubling the
          number of paths only reduces error by ~29%. This is why efficient algorithms matter!
        </p>
      </div>
    </motion.div>
  );
};

export default MonteCarloConvergence;
