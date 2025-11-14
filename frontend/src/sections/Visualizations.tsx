import { motion } from 'framer-motion';

const Visualizations = () => {
  return (
    <section id="visualizations" className="section bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Visualizations
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Explore option pricing through interactive visualizations
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Placeholder viz cards */}
            {[
              '3D Surface Plot',
              'Monte Carlo Convergence',
              'Algorithm Comparison',
              'Backward Induction Animation',
            ].map((title, idx) => (
              <div key={idx} className="glass rounded-2xl p-6 hover-lift">
                <h3 className="text-xl font-semibold mb-4 text-gray-800">
                  {title}
                </h3>
                <div className="h-64 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg flex items-center justify-center">
                  <p className="text-gray-500">Coming soon...</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Visualizations;
