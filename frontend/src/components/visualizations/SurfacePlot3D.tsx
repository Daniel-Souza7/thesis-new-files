import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import axios from 'axios';

interface SurfacePlot3DProps {
  algorithm?: string;
  params?: any;
}

const SurfacePlot3D = ({ algorithm = 'RLSM', params }: SurfacePlot3DProps) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Generate synthetic surface data for demonstration
  useEffect(() => {
    generateSurfaceData();
  }, [algorithm, params]);

  const generateSurfaceData = () => {
    setLoading(true);

    // Create a grid of stock prices and times
    const stockPrices = Array.from({ length: 30 }, (_, i) => 80 + i * 2); // 80 to 138
    const times = Array.from({ length: 20 }, (_, i) => i * 0.05); // 0 to 0.95

    // Generate option values (synthetic data for demo)
    const optionValues = times.map((t) =>
      stockPrices.map((S) => {
        const K = 100; // strike price
        const intrinsic = Math.max(S - K, 0);
        const timeValue = intrinsic * Math.exp(-0.05 * (1 - t)) * (1 - t) * 0.3;
        return intrinsic + timeValue + Math.random() * 2;
      })
    );

    setData({
      x: stockPrices,
      y: times,
      z: optionValues,
      type: 'surface',
      colorscale: 'Viridis',
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: '#42f462',
          project: { z: true },
        },
      },
    });

    setLoading(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full h-full bg-white rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-4">
        Option Value Surface - {algorithm}
      </h3>
      <p className="text-gray-600 mb-4">
        3D visualization of American call option values across different stock prices and time to maturity
      </p>

      {/* Option specification */}
      <div className="mb-4 p-3 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
        <p className="text-sm text-gray-700">
          <strong>Option:</strong> American Call on single asset | <strong>Strike:</strong> K=$100 |
          <strong> Payoff:</strong> max(S - 100, 0) | <strong>Risk-free rate:</strong> r=5%
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : data ? (
        <Plot
          data={[data]}
          layout={{
            autosize: true,
            height: 500,
            scene: {
              xaxis: { title: 'Stock Price ($)' },
              yaxis: { title: 'Time to Maturity' },
              zaxis: { title: 'Option Value ($)' },
              camera: {
                eye: { x: 1.5, y: 1.5, z: 1.3 },
              },
            },
            margin: { l: 0, r: 0, b: 0, t: 0 },
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
          }}
          style={{ width: '100%', height: '500px' }}
        />
      ) : (
        <div className="text-center text-gray-500">No data available</div>
      )}

      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-gray-700">
          <strong>Tip:</strong> Drag to rotate, scroll to zoom, hover over the surface to see values
        </p>
      </div>
    </motion.div>
  );
};

export default SurfacePlot3D;
