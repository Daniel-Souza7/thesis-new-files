import { motion } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { useState } from 'react';

const Problem = () => {
  const [hoveredTerm, setHoveredTerm] = useState<string | null>(null);

  const termDefinitions: Record<string, string> = {
    'V_t': 'Option value at time t (what we want to find)',
    'S_t': 'Stock price(s) at time t',
    'g': 'Immediate exercise payoff (intrinsic value)',
    'E': 'Expected continuation value if we wait',
    'r': 'Risk-free interest rate',
    'Delta_t': 'Time step size',
  };

  return (
    <section id="problem" className="section bg-white">
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
              The Problem
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Pricing American options requires solving an optimal stopping problem
            </p>
          </div>

          {/* Problem description */}
          <div className="grid md:grid-cols-2 gap-12 mb-16">
            <div>
              <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                What is an American Option?
              </h3>
              <p className="text-gray-600 leading-relaxed mb-4">
                Unlike European options that can only be exercised at maturity,
                <strong className="text-gray-900"> American options</strong> can be
                exercised at <strong className="text-gray-900">any time</strong> before expiration.
              </p>
              <p className="text-gray-600 leading-relaxed">
                This flexibility makes them more valuable but significantly harder to price.
                We must determine <strong className="text-gray-900">when</strong> to exercise
                to maximize value.
              </p>
            </div>

            <div>
              <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                The Challenge
              </h3>
              <p className="text-gray-600 leading-relaxed mb-4">
                At each time step, the holder faces a decision:
              </p>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-start">
                  <span className="text-green-600 mr-2">•</span>
                  <span><strong>Exercise now:</strong> Get immediate payoff <InlineMath math="g(S_t)" /></span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>Wait:</strong> Keep the option alive, hoping for better value</span>
                </li>
              </ul>
              <p className="text-gray-600 leading-relaxed mt-4">
                Finding the optimal exercise strategy requires backward induction and
                estimating continuation values.
              </p>
            </div>
          </div>

          {/* Bellman Equation */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            viewport={{ once: true }}
            className="glass rounded-2xl p-8 mb-12"
          >
            <h3 className="text-2xl font-semibold mb-6 text-center text-gray-800">
              Bellman Equation for Optimal Stopping
            </h3>

            <div className="bg-blue-50 rounded-xl p-6 mb-6">
              <BlockMath math="V_t(S_t) = \max\left\{g(S_t), \, \mathbb{E}\left[e^{-r\Delta t}V_{t+1}(S_{t+1}) \mid S_t\right]\right\}" />
            </div>

            <p className="text-center text-gray-600 mb-6">
              Hover over terms below to see their meaning:
            </p>

            {/* Interactive term explanations */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(termDefinitions).map(([term, definition]) => (
                <motion.div
                  key={term}
                  className="relative"
                  onHoverStart={() => setHoveredTerm(term)}
                  onHoverEnd={() => setHoveredTerm(null)}
                >
                  <div className="p-4 bg-white rounded-lg border-2 border-gray-200 hover:border-blue-400 transition-colors cursor-help">
                    <div className="text-center mb-2">
                      <InlineMath math={term} />
                    </div>

                    {/* Tooltip */}
                    {hoveredTerm === term && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64"
                      >
                        <div className="bg-gray-900 text-white text-sm rounded-lg p-3 shadow-xl">
                          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-gray-900" />
                          {definition}
                        </div>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* The Challenge: Computing Continuation Values */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            viewport={{ once: true }}
            className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-8"
          >
            <h3 className="text-2xl font-semibold mb-4 text-gray-800">
              The Core Challenge
            </h3>
            <p className="text-gray-700 leading-relaxed mb-4">
              The key difficulty is estimating the <strong>continuation value</strong>:
            </p>
            <div className="bg-white rounded-xl p-6 mb-4">
              <BlockMath math="C_t(S_t) = \mathbb{E}\left[e^{-r\Delta t}V_{t+1}(S_{t+1}) \mid S_t\right]" />
            </div>
            <p className="text-gray-700 leading-relaxed mb-4">
              This is an <strong>infinite-dimensional problem</strong> - we need to approximate
              this function for all possible stock prices. Traditional approaches:
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-blue-600 mb-2">Polynomial Basis</h4>
                <p className="text-sm text-gray-600">LSM uses degree-2 polynomials (Longstaff-Schwartz, 2001)</p>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-green-600 mb-2">Neural Networks</h4>
                <p className="text-sm text-gray-600">NLSM trains networks via gradient descent (Lapeyre-Lelong, 2019)</p>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-purple-600 mb-2">Randomized Networks</h4>
                <p className="text-sm text-gray-600"><strong>RLSM (our focus)</strong> uses frozen random weights (Herrera et al., 2021)</p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Problem;
