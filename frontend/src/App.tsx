import Navigation from './components/Navigation';
import Hero from './sections/Hero';
import Problem from './sections/Problem';
import Algorithms from './sections/Algorithms';
import CodeExamples from './sections/CodeExamples';
import Demo from './sections/Demo';
import Visualizations from './sections/Visualizations';
import Results from './sections/Results';

function App() {
  return (
    <div className="App">
      <Navigation />
      <Hero />
      <Problem />
      <Algorithms />
      <CodeExamples />
      <Demo />
      <Visualizations />
      <Results />

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <p className="text-lg font-medium mb-2">
            Optimal Stopping Explorer
          </p>
          <p className="text-gray-400 mb-4">
            MSc Thesis · Quantitative Methods in Finance
          </p>
          <p className="text-gray-500">
            Daniel Souza · University of Coimbra · 2025
          </p>
          <div className="mt-6 pt-6 border-t border-gray-800">
            <p className="text-sm text-gray-400">
              Built with React, TypeScript, Flask, and ❤️
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
