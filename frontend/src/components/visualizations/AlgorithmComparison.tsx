import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';

interface ComparisonData {
  algorithm: string;
  price: number;
  time: number;
  accuracy: number;
}

const AlgorithmComparison = () => {
  const [viewMode, setViewMode] = useState<'price' | 'time' | 'accuracy'>('price');
  const [data, setData] = useState<ComparisonData[]>([]);

  useEffect(() => {
    // Generate synthetic comparison data
    // Problem: 10-dimensional Max Call option, S0=100, K=100, T=1, σ=0.2
    const comparisonData: ComparisonData[] = [
      { algorithm: 'RLSM', price: 11.28, time: 0.18, accuracy: 99.1 },
      { algorithm: 'RFQI', price: 11.25, time: 0.16, accuracy: 98.9 },
      { algorithm: 'LSM', price: 11.32, time: 1.45, accuracy: 99.4 },
      { algorithm: 'NLSM', price: 11.31, time: 2.87, accuracy: 99.3 },
      { algorithm: 'DOS', price: 11.29, time: 8.42, accuracy: 99.6 },
      { algorithm: 'FQI', price: 11.24, time: 3.15, accuracy: 98.7 },
    ];
    setData(comparisonData);
  }, []);

  const renderChart = () => {
    switch (viewMode) {
      case 'price':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="algorithm" />
              <YAxis label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="price" fill="#3b82f6" name="Option Price" />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'time':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="algorithm" />
              <YAxis label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="time" fill="#10b981" name="Computation Time" />
            </BarChart>
          </ResponsiveContainer>
        );
      case 'accuracy':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="algorithm" />
              <YAxis
                domain={[95, 100]}
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#f59e0b"
                strokeWidth={3}
                name="Accuracy"
                dot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-4">Algorithm Performance Comparison</h3>
      <p className="text-gray-600 mb-4">
        Compare pricing accuracy, computation time, and results across all algorithms
      </p>

      {/* Problem description */}
      <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-gray-900 mb-2">Test Problem</h4>
        <div className="text-sm text-gray-700 space-y-1">
          <p><strong>Payoff:</strong> Max Call option - max(max(S₁, S₂, ..., S₁₀) - K, 0)</p>
          <p><strong>Dimension:</strong> 10 assets</p>
          <p><strong>Parameters:</strong> S₀ = $100, K = $100, T = 1 year, σ = 20%, r = 5%</p>
          <p><strong>Simulation:</strong> 10,000 paths, 10 time steps</p>
        </div>
      </div>

      {/* View mode selector */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setViewMode('price')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'price'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Option Prices
        </button>
        <button
          onClick={() => setViewMode('time')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'time'
              ? 'bg-green-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Computation Time
        </button>
        <button
          onClick={() => setViewMode('accuracy')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            viewMode === 'accuracy'
              ? 'bg-amber-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Accuracy
        </button>
      </div>

      {renderChart()}

      {/* Summary table */}
      <div className="mt-6 overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Algorithm
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Price ($)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Time (s)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Accuracy (%)
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((row) => (
              <tr key={row.algorithm} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                  {row.algorithm}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                  ${row.price.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                  {row.time.toFixed(2)}s
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                  {row.accuracy.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};

export default AlgorithmComparison;
