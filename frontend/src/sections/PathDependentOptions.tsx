import { motion } from 'framer-motion';
import { AlertTriangle, TrendingUp, BarChart3 } from 'lucide-react';
import DoubleBarrierVisualization from '../components/visualizations/DoubleBarrierVisualization';

const PathDependentOptions = () => {
  return (
    <section id="path-dependent" className="section bg-gradient-to-br from-orange-50 to-amber-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          {/* Header */}
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Path-Dependent Options
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              When the entire price history matters, not just the final value
            </p>
          </div>

          {/* Explanation Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-8 h-8 text-orange-600" />
                <h3 className="text-xl font-semibold text-gray-800">Barrier Options</h3>
              </div>
              <p className="text-gray-700 mb-3">
                Option value depends on whether the stock price crosses certain levels during
                the option's lifetime.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• <strong>Knock-Out:</strong> Dies if barrier hit</li>
                <li>• <strong>Knock-In:</strong> Activates if barrier hit</li>
                <li>• <strong>Double Barrier:</strong> Two boundaries</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-8 h-8 text-orange-600" />
                <h3 className="text-xl font-semibold text-gray-800">Lookback Options</h3>
              </div>
              <p className="text-gray-700 mb-3">
                Payoff depends on the maximum or minimum price reached during the option's life.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• <strong>Floating Strike:</strong> Strike = min/max price</li>
                <li>• <strong>Fixed Strike:</strong> Payoff uses extreme value</li>
                <li>• Requires tracking price history</li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <BarChart3 className="w-8 h-8 text-orange-600" />
                <h3 className="text-xl font-semibold text-gray-800">Asian Options</h3>
              </div>
              <p className="text-gray-700 mb-3">
                Payoff based on the average price of the underlying asset over a period.
              </p>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• <strong>Arithmetic Average:</strong> Simple average</li>
                <li>• <strong>Geometric Average:</strong> Log-returns average</li>
                <li>• Reduces volatility impact</li>
              </ul>
            </motion.div>
          </div>

          {/* The Challenge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="mb-12 p-6 bg-white rounded-2xl shadow-lg"
          >
            <h3 className="text-2xl font-bold text-gray-900 mb-4">The Challenge</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-red-600 mb-2">❌ Standard Algorithms Fail</h4>
                <p className="text-gray-700 mb-2">
                  RLSM, RFQI, LSM, NLSM, DOS, FQI assume <strong>Markovian dynamics</strong>:
                </p>
                <ul className="text-sm text-gray-600 space-y-1 ml-4">
                  <li>• Future only depends on current state S_t</li>
                  <li>• Cannot remember if barrier was hit</li>
                  <li>• Cannot track running max/min/average</li>
                  <li>• Miss critical path information</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-green-600 mb-2">✅ Extended Algorithms Work</h4>
                <p className="text-gray-700 mb-2">
                  <strong>SRLSM</strong> and <strong>SRFQI</strong> use <strong>Recurrent Neural Networks</strong>:
                </p>
                <ul className="text-sm text-gray-600 space-y-1 ml-4">
                  <li>• Hidden state h_t remembers history</li>
                  <li>• Can track if barriers were breached</li>
                  <li>• Can maintain running statistics</li>
                  <li>• Handle non-Markovian problems</li>
                </ul>
              </div>
            </div>
          </motion.div>

          {/* Interactive Visualization */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="mb-12"
          >
            <DoubleBarrierVisualization />
          </motion.div>

          {/* Algorithm Extensions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="p-6 bg-white rounded-2xl shadow-lg"
          >
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Algorithm Extensions for Path Dependence
            </h3>

            <div className="grid md:grid-cols-2 gap-6">
              {/* SRLSM */}
              <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                <h4 className="text-lg font-semibold text-gray-900 mb-2">
                  SRLSM (Stateful RLSM)
                </h4>
                <p className="text-sm text-gray-700 mb-3">
                  Extends RLSM with recurrent neural networks to handle path-dependent payoffs.
                </p>
                <div className="font-mono text-xs bg-gray-900 text-green-400 p-3 rounded">
                  {"h_t = tanh(W_h * [S_t, h_{t-1}])"}<br />
                  {"features = RNN(S_{0:t})"}
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  The hidden state h_t carries information from all previous time steps.
                </p>
              </div>

              {/* SRFQI */}
              <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                <h4 className="text-lg font-semibold text-gray-900 mb-2">
                  SRFQI (Stateful RFQI)
                </h4>
                <p className="text-sm text-gray-700 mb-3">
                  Extends RFQI with recurrent Q-iteration to learn optimal stopping with memory.
                </p>
                <div className="font-mono text-xs bg-gray-900 text-green-400 p-3 rounded">
                  {"Q_t(s_t, h_t) = max(payoff, E[Q_{t+1}])"}<br />
                  {"state = [S_t, h_t, barrier_hit]"}
                </div>
                <p className="text-xs text-gray-600 mt-2">
                  Explicitly tracks path-dependent features in the state representation.
                </p>
              </div>
            </div>

            <div className="mt-6 p-4 bg-amber-50 rounded-lg border border-amber-200">
              <p className="text-sm text-gray-700">
                <strong>Key Insight:</strong> By adding recurrent connections, the algorithms can
                "remember" critical events (like barrier breaches, running extremes, or averages)
                that standard feedforward approaches would miss. This makes pricing path-dependent
                options feasible with neural network methods.
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default PathDependentOptions;
