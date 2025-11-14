import { motion } from 'framer-motion';
import SurfacePlot3D from '../components/visualizations/SurfacePlot3D';
import AlgorithmComparison from '../components/visualizations/AlgorithmComparison';
import MonteCarloConvergence from '../components/visualizations/MonteCarloConvergence';
import BackwardInduction from '../components/visualizations/BackwardInduction';
import PathBreakdown from '../components/visualizations/PathBreakdown';

const Visualizations = () => {
  return (
    <section id="visualizations" className="section bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Interactive Visualizations
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Explore option pricing algorithms through interactive charts and animations
            </p>
          </div>

          <div className="space-y-12">
            {/* 3D Surface Plot */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <SurfacePlot3D />
            </motion.div>

            {/* Algorithm Comparison */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              viewport={{ once: true }}
            >
              <AlgorithmComparison />
            </motion.div>

            {/* Monte Carlo Convergence */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <MonteCarloConvergence />
            </motion.div>

            {/* Backward Induction Animation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              viewport={{ once: true }}
            >
              <BackwardInduction />
            </motion.div>

            {/* Path Breakdown */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <PathBreakdown />
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Visualizations;
