import { motion } from 'framer-motion';

const Results = () => {
  return (
    <section id="results" className="section bg-gradient-to-br from-gray-50 via-white to-blue-50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Results & Conclusions
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Key findings from algorithm comparisons
            </p>
          </div>

          <div className="space-y-8">
            {/* Performance summary */}
            <div className="glass rounded-2xl p-8">
              <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                Performance Summary
              </h3>
              <p className="text-gray-600">
                Comparative analysis of RLSM, RFQI, and benchmark algorithms on various problem sizes.
              </p>
            </div>

            {/* Key insights */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl p-6 border-2 border-blue-200">
                <h4 className="text-lg font-semibold mb-2 text-blue-600">Speed</h4>
                <p className="text-gray-600">RLSM is 5-10x faster than NLSM/DOS</p>
              </div>
              <div className="bg-white rounded-xl p-6 border-2 border-green-200">
                <h4 className="text-lg font-semibold mb-2 text-green-600">Accuracy</h4>
                <p className="text-gray-600">Comparable to benchmark algorithms</p>
              </div>
              <div className="bg-white rounded-xl p-6 border-2 border-purple-200">
                <h4 className="text-lg font-semibold mb-2 text-purple-600">Scalability</h4>
                <p className="text-gray-600">Handles high-dimensional problems</p>
              </div>
              <div className="bg-white rounded-xl p-6 border-2 border-orange-200">
                <h4 className="text-lg font-semibold mb-2 text-orange-600">Stability</h4>
                <p className="text-gray-600">No gradient descent, more robust</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Results;
