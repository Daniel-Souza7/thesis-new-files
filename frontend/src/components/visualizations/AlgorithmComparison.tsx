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
    const comparisonData: ComparisonData[] = [
      { algorithm: 'RLSM', price: 11.25, time: 0.15, accuracy: 98.5 },
      { algorithm: 'RFQI', price: 11.18, time: 0.22, accuracy: 97.8 },
      { algorithm: 'LSM', price: 11.32, time: 2.45, accuracy: 99.2 },
      { algorithm: 'NLSM', price: 11.28, time: 3.12, accuracy: 98.9 },
      { algorithm: 'DOS', price: 11.26, time: 5.67, accuracy: 99.5 },
      { algorithm: 'FQI', price: 11.21, time: 4.33, accuracy: 98.1 },
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
