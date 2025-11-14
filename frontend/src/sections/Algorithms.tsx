import { motion } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { useState } from 'react';
import type { Algorithm } from '../types';

const algorithmData: Record<Algorithm, {
  fullName: string;
  description: string;
  formula: string;
  keyPoints: string[];
  reference: string;
}> = {
  RLSM: {
    fullName: 'Randomized Least Squares Monte Carlo',
    description: 'Uses randomized neural networks with frozen weights as basis functions for least squares regression.',
    formula: 'C_t(S_t) \\approx \\sum_{i=1}^{H} \\beta_i \\cdot \\phi(W_i^T S_t)',
    keyPoints: [
      'Random weights W_i are frozen (not trained)',
      'Only output coefficients β_i learned via least squares',
      'Fast and stable - no gradient descent needed',
      'Scales well to high dimensions',
    ],
    reference: 'Herrera, Krach, Ruyssen, Teichmann (2021)',
  },
  RFQI: {
    fullName: 'Randomized Fitted Q-Iteration',
    description: 'Q-learning approach using randomized neural networks for basis functions.',
    formula: 'Q_t(S_t, a) \\approx \\sum_{i=1}^{H} \\beta_i \\cdot \\phi(W_i^T S_t)',
    keyPoints: [
      'Iteratively refines Q-values',
      'Randomized basis functions',
      'Handles path-dependent options',
      'Robust to parameter choices',
    ],
    reference: 'Herrera, Krach, Ruyssen, Teichmann (2021)',
  },
  LSM: {
    fullName: 'Least Squares Monte Carlo',
    description: 'Classic algorithm using polynomial basis functions for regression.',
    formula: 'C_t(S_t) \\approx \\sum_{i=0}^{d} \\beta_i \\cdot p_i(S_t)',
    keyPoints: [
      'Uses degree-2 polynomials',
      'Industry standard benchmark',
      'Simple and interpretable',
      'May struggle in high dimensions',
    ],
    reference: 'Longstaff & Schwartz (2001)',
  },
  NLSM: {
    fullName: 'Neural Least Squares Monte Carlo',
    description: 'Fully trained neural networks to approximate continuation values.',
    formula: 'C_t(S_t) \\approx NN(S_t; \\theta)',
    keyPoints: [
      'All weights trained via gradient descent',
      'Flexible function approximator',
      'Can capture complex patterns',
      'Slower than RLSM due to training',
    ],
    reference: 'Lapeyre & Lelong (2019)',
  },
  DOS: {
    fullName: 'Deep Optimal Stopping',
    description: 'Neural network learns stopping decisions directly, not continuation values.',
    formula: '\\pi_t(S_t) \\approx NN(S_t; \\theta)',
    keyPoints: [
      'Learns stopping probability',
      'Maximizes expected payoff',
      'End-to-end optimization',
      'Different paradigm from LSM',
    ],
    reference: 'Becker, Cheridito, Jentzen (2019)',
  },
  FQI: {
    fullName: 'Fitted Q-Iteration',
    description: 'Q-learning with polynomial basis functions.',
    formula: 'Q_t(S_t, a) \\approx \\sum_{i=0}^{d} \\beta_i \\cdot p_i(S_t)',
    keyPoints: [
      'Iterative Q-value refinement',
      'Polynomial basis functions',
      'Works for general RL problems',
      'Includes time as a feature',
    ],
    reference: 'Tsitsiklis & Van Roy (2001)',
  },
};

const Algorithms = () => {
  const [selectedAlgo, setSelectedAlgo] = useState<Algorithm>('RLSM');
  const data = algorithmData[selectedAlgo];

  return (
    <section id="algorithms" className="section bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          {/* Section header */}
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Algorithms
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Compare different approaches to solving the American option pricing problem
            </p>
          </div>

          {/* Algorithm selector */}
          <div className="flex flex-wrap justify-center gap-3 mb-12">
            {(Object.keys(algorithmData) as Algorithm[]).map((algo) => (
              <button
                key={algo}
                onClick={() => setSelectedAlgo(algo)}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  selectedAlgo === algo
                    ? 'bg-blue-600 text-white shadow-lg'
                    : 'bg-white text-gray-700 hover:bg-gray-50 border-2 border-gray-200'
                }`}
              >
                {algo}
              </button>
            ))}
          </div>

          {/* Algorithm details */}
          <motion.div
            key={selectedAlgo}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="glass rounded-2xl p-8"
          >
            <h3 className="text-3xl font-bold mb-4 text-gray-900">
              {data.fullName}
            </h3>
            <p className="text-lg text-gray-600 mb-8">
              {data.description}
            </p>

            {/* Formula */}
            <div className="bg-blue-50 rounded-xl p-6 mb-8">
              <p className="text-sm font-medium text-gray-700 mb-2">Key Formula:</p>
              <BlockMath math={data.formula} />
            </div>

            {/* Key points */}
            <div className="mb-6">
              <h4 className="text-xl font-semibold mb-4 text-gray-800">Key Features:</h4>
              <ul className="grid md:grid-cols-2 gap-3">
                {data.keyPoints.map((point, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="text-blue-600 mr-2 mt-1">✓</span>
                    <span className="text-gray-700">{point}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Reference */}
            <div className="text-sm text-gray-500 border-t border-gray-200 pt-4">
              <strong>Reference:</strong> {data.reference}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Algorithms;
