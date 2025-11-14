import { motion } from 'framer-motion';
import { useState } from 'react';
import { Play } from 'lucide-react';

const Demo = () => {
  const [nbStocks, setNbStocks] = useState(5);
  const [volatility, setVolatility] = useState(0.2);
  const [maturity, setMaturity] = useState(1.0);

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
            <div className="glass rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-6 text-gray-800">
                Parameters
              </h3>

              <div className="space-y-6">
                {/* Number of stocks */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Stocks: {nbStocks}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={nbStocks}
                    onChange={(e) => setNbStocks(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Volatility */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Volatility: {(volatility * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={volatility}
                    onChange={(e) => setVolatility(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Maturity */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Maturity: {maturity.toFixed(2)} years
                  </label>
                  <input
                    type="range"
                    min="0.25"
                    max="2"
                    step="0.25"
                    value={maturity}
                    onChange={(e) => setMaturity(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                <button className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">
                  <Play size={20} />
                  Run Pricing
                </button>
              </div>
            </div>

            {/* Results (placeholder) */}
            <div className="lg:col-span-2 glass rounded-2xl p-6">
              <h3 className="text-xl font-semibold mb-6 text-gray-800">
                Results
              </h3>
              <div className="h-64 flex items-center justify-center bg-gray-100 rounded-lg">
                <p className="text-gray-500">
                  Click "Run Pricing" to see results
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Demo;
