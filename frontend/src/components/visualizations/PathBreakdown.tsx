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
  Scatter,
  ScatterChart,
  ZAxis,
} from 'recharts';

interface PathPoint {
  time: number;
  price: number;
}

interface PathData {
  id: number;
  points: PathPoint[];
  exerciseTime: number | null;
  exercisePrice: number | null;
  payoff: number;
}

const PathBreakdown = () => {
  const [paths, setPaths] = useState<PathData[]>([]);
  const [selectedPath, setSelectedPath] = useState<number>(0);
  const [showAll, setShowAll] = useState(true);

  useEffect(() => {
    generatePaths();
  }, []);

  const generatePaths = () => {
    const numPaths = 10;
    const numSteps = 11;
    const S0 = 100;
    const K = 100;
    const T = 1;
    const sigma = 0.2;
    const dt = T / (numSteps - 1);

    const generatedPaths: PathData[] = [];

    for (let p = 0; p < numPaths; p++) {
      const points: PathPoint[] = [];
      let S = S0;

      for (let i = 0; i < numSteps; i++) {
        points.push({ time: i * dt, price: S });
        if (i < numSteps - 1) {
          const z = Math.random() * 2 - 1; // Simple random walk
          S = S * Math.exp(sigma * Math.sqrt(dt) * z);
        }
      }

      // Determine exercise time (simplified logic)
      let exerciseTime: number | null = null;
      let exercisePrice: number | null = null;
      let payoff = 0;

      for (let i = points.length - 1; i >= 0; i--) {
        const immediate = Math.max(K - points[i].price, 0);
        if (immediate > 5 && Math.random() > 0.5) {
          exerciseTime = points[i].time;
          exercisePrice = points[i].price;
          payoff = immediate;
          break;
        }
      }

      if (exerciseTime === null) {
        const finalPrice = points[points.length - 1].price;
        exerciseTime = T;
        exercisePrice = finalPrice;
        payoff = Math.max(K - finalPrice, 0);
      }

      generatedPaths.push({
        id: p,
        points,
        exerciseTime,
        exercisePrice,
        payoff,
      });
    }

    setPaths(generatedPaths);
  };

  const currentPath = paths[selectedPath];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-white rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-2">Monte Carlo Path Analysis</h3>
      <p className="text-gray-600 mb-4">
        Examine individual simulation paths and exercise decisions
      </p>

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
            {paths.map((path) => (
              <option key={path.id} value={path.id}>
                Path {path.id + 1} - Payoff: ${path.payoff.toFixed(2)}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end">
          <button
            onClick={() => setShowAll(!showAll)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              showAll
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {showAll ? 'Show Selected Only' : 'Show All Paths'}
          </button>
        </div>
      </div>

      {currentPath && (
        <>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="time"
                type="number"
                domain={[0, 1]}
                label={{ value: 'Time (years)', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                domain={[80, 120]}
                label={{ value: 'Stock Price ($)', angle: -90, position: 'insideLeft' }}
                tickFormatter={(value) => `$${Math.round(value)}`}
              />
              <Tooltip
                formatter={(value: number) => [`$${Math.round(value as number)}`, 'Price']}
                labelFormatter={(label) => `Time: ${label.toFixed(2)}y`}
              />
              <Legend />
              <ReferenceLine
                y={100}
                stroke="#64748b"
                strokeDasharray="5 5"
                label={{ value: 'Strike K=100', position: 'right' }}
              />

              {/* Show all paths if enabled */}
              {showAll &&
                paths.map((path) =>
                  path.id !== selectedPath ? (
                    <Line
                      key={path.id}
                      data={path.points}
                      type="monotone"
                      dataKey="price"
                      stroke="#cbd5e1"
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ) : null
                )}

              {/* Selected path */}
              <Line
                data={currentPath.points}
                type="monotone"
                dataKey="price"
                stroke="#3b82f6"
                strokeWidth={3}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  const isExercisePoint =
                    currentPath.exerciseTime !== null &&
                    Math.abs(payload.time - currentPath.exerciseTime) < 0.01;

                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isExercisePoint ? 7 : 4}
                      fill={isExercisePoint ? '#ef4444' : '#3b82f6'}
                      stroke={isExercisePoint ? '#dc2626' : '#3b82f6'}
                      strokeWidth={isExercisePoint ? 2 : 1}
                    />
                  );
                }}
                name="Stock Price Path"
              />

              {/* Exercise point */}
              {currentPath.exerciseTime !== null && (
                <Scatter
                  data={[
                    {
                      time: currentPath.exerciseTime,
                      price: currentPath.exercisePrice,
                    },
                  ]}
                  fill="#ef4444"
                  shape="star"
                  name="Exercise Point"
                />
              )}
            </LineChart>
          </ResponsiveContainer>

          {/* Path statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Initial Price</p>
              <p className="text-xl font-bold text-blue-600">
                ${currentPath.points[0].price.toFixed(2)}
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Final Price</p>
              <p className="text-xl font-bold text-green-600">
                ${currentPath.points[currentPath.points.length - 1].price.toFixed(2)}
              </p>
            </div>
            <div className="p-4 bg-amber-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Exercise Time</p>
              <p className="text-xl font-bold text-amber-600">
                {currentPath.exerciseTime?.toFixed(2)}y
              </p>
            </div>
            <div className="p-4 bg-red-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Payoff</p>
              <p className="text-xl font-bold text-red-600">
                ${currentPath.payoff.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Path details */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-2">Path Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-600">Strike Price:</span>
                <span className="ml-2 font-medium">$100.00</span>
              </div>
              <div>
                <span className="text-gray-600">Exercise Decision:</span>
                <span className="ml-2 font-medium">
                  {currentPath.exerciseTime === 1.0 ? 'Hold to maturity' : 'Early exercise'}
                </span>
              </div>
              <div>
                <span className="text-gray-600">Intrinsic Value at Exercise:</span>
                <span className="ml-2 font-medium">
                  ${Math.max(100 - (currentPath.exercisePrice || 0), 0).toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-600">In/Out of Money:</span>
                <span className="ml-2 font-medium">
                  {(currentPath.exercisePrice || 0) < 100 ? 'In the money' : 'Out of money'}
                </span>
              </div>
            </div>
          </div>
        </>
      )}

      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-gray-700">
          <strong>Interpretation:</strong> Each path represents one possible future evolution of the
          stock price. The algorithm must decide the optimal exercise time for each path. The option
          value is the average discounted payoff across all paths.
        </p>
      </div>
    </motion.div>
  );
};

export default PathBreakdown;
