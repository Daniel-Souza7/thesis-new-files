import { motion } from 'framer-motion';
import { useState } from 'react';
import CodeEditor from '../components/CodeEditor';
import type { Algorithm } from '../types';

const CodeExamples = () => {
  const [selectedAlgo, setSelectedAlgo] = useState<Algorithm>('RLSM');

  const codeExamples: Record<Algorithm, string> = {
    RLSM: `# Randomized Least Squares Monte Carlo
import numpy as np
import torch.nn as nn

class RLSM:
    """American option pricing using random features."""

    def __init__(self, hidden_size=100):
        self.hidden_size = hidden_size
        # Random features (frozen, never trained)
        self.W = np.random.randn(hidden_size, n_stocks)
        self.b = np.random.randn(hidden_size)

    def basis_functions(self, X):
        """Apply random feature transformation."""
        # X: (n_paths, n_stocks)
        Z = X @ self.W.T + self.b  # (n_paths, hidden_size)
        return np.tanh(Z)  # Activation function

    def price(self, stock_paths, payoffs):
        """Backward induction with random features."""
        n_paths, n_stocks, n_dates = stock_paths.shape

        # Initialize with payoffs at maturity
        continuation_value = payoffs[:, -1].copy()

        # Backward induction
        for t in range(n_dates - 1, 0, -1):
            # Discount continuation values
            continuation_value = np.exp(-r * dt) * continuation_value

            # Get current stock prices
            S_t = stock_paths[:, :, t]

            # Immediate exercise value
            immediate = payoffs[:, t]

            # Find ITM paths
            itm = immediate > 0

            if itm.sum() > 0:
                # Transform to feature space
                Phi = self.basis_functions(S_t[itm])

                # Least squares regression (only trainable part!)
                beta = np.linalg.lstsq(Phi, continuation_value[itm], rcond=None)[0]

                # Predict continuation value
                predicted = Phi @ beta

                # Exercise decision
                exercise = immediate[itm] >= predicted
                continuation_value[itm] = np.where(
                    exercise,
                    immediate[itm],
                    continuation_value[itm]
                )

        # Final discounting to t=0
        continuation_value = np.exp(-r * dt) * continuation_value
        option_value = np.maximum(payoffs[:, 0], continuation_value)

        return np.mean(option_value)`,

    RFQI: `# Randomized Fitted Q-Iteration
import numpy as np

class RFQI:
    """American option pricing using Q-learning with random features."""

    def __init__(self, hidden_size=100):
        self.hidden_size = hidden_size
        # Random features (frozen)
        self.W = np.random.randn(hidden_size, n_stocks + 1)
        self.b = np.random.randn(hidden_size)

    def feature_map(self, X):
        """Random feature transformation."""
        return np.tanh(X @ self.W.T + self.b)

    def price(self, stock_paths, payoffs):
        """Q-iteration with random features."""
        n_paths, n_stocks, n_dates = stock_paths.shape

        # Initialize Q-values with terminal payoffs
        Q_values = payoffs[:, -1].copy()

        # Backward iteration
        for t in range(n_dates - 1, 0, -1):
            # Discount Q-values
            Q_values = np.exp(-r * dt) * Q_values

            # Augment state with time
            time_feature = np.full((n_paths, 1), t / n_dates)
            state = np.hstack([stock_paths[:, :, t], time_feature])

            # Transform to feature space
            Phi = self.feature_map(state)

            # Fit Q-function via least squares
            theta = np.linalg.lstsq(Phi, Q_values, rcond=None)[0]

            # Predict Q-values (continuation)
            Q_continue = Phi @ theta

            # Compare with immediate exercise
            Q_exercise = payoffs[:, t]

            # Update Q-values (take maximum)
            Q_values = np.maximum(Q_exercise, Q_continue)

        # Price at t=0
        time_0 = np.zeros((n_paths, 1))
        state_0 = np.hstack([stock_paths[:, :, 0], time_0])
        Phi_0 = self.feature_map(state_0)
        Q_0 = Phi_0 @ theta

        return np.mean(np.maximum(payoffs[:, 0], Q_0))`,

    LSM: `# Longstaff-Schwartz Monte Carlo
import numpy as np

class LSM:
    """Classic American option pricing algorithm."""

    def basis_functions(self, S):
        """Polynomial basis functions."""
        # S: (n_paths, n_stocks)
        # Use Laguerre polynomials up to degree 3
        basis = []

        for i in range(S.shape[1]):
            s = S[:, i]
            basis.append(np.ones_like(s))           # 1
            basis.append(s)                          # s
            basis.append(1 - s)                      # 1 - s
            basis.append(0.5 * (2 - 4*s + s**2))    # (2 - 4s + s²)/2
            basis.append(1.0/6 * (6 - 18*s + 9*s**2 - s**3))  # ...

        return np.column_stack(basis)

    def price(self, stock_paths, payoffs):
        """Backward induction with polynomial regression."""
        n_paths, n_stocks, n_dates = stock_paths.shape

        # Initialize with terminal payoffs
        continuation_value = payoffs[:, -1].copy()

        # Backward induction
        for t in range(n_dates - 1, 0, -1):
            # Discount continuation values
            continuation_value = np.exp(-r * dt) * continuation_value

            # Get current stock prices
            S_t = stock_paths[:, :, t]
            immediate = payoffs[:, t]

            # Regression only on ITM paths
            itm = immediate > 1e-6

            if itm.sum() > 10:  # Need enough paths
                # Construct basis functions
                Phi = self.basis_functions(S_t[itm])

                # Least squares regression
                beta = np.linalg.lstsq(
                    Phi,
                    continuation_value[itm],
                    rcond=None
                )[0]

                # Predict continuation value
                predicted = Phi @ beta

                # Exercise decision
                exercise = immediate[itm] >= predicted
                continuation_value[itm] = np.where(
                    exercise,
                    immediate[itm],
                    continuation_value[itm]
                )

        # Final step
        continuation_value = np.exp(-r * dt) * continuation_value
        option_value = np.maximum(payoffs[:, 0], continuation_value)

        return np.mean(option_value)`,

    NLSM: `# Neural Least Squares Monte Carlo
import torch
import torch.nn as nn

class NLSM:
    """LSM with neural network basis functions."""

    def __init__(self, hidden_size=64):
        self.network = nn.Sequential(
            nn.Linear(n_stocks, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Initialize network weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def train_network(self, X, y):
        """Train neural network features."""
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        for epoch in range(100):
            optimizer.zero_grad()
            features = self.network(X_tensor)

            # Linear regression in feature space
            beta = torch.linalg.lstsq(features, y_tensor).solution
            pred = features @ beta

            loss = ((pred - y_tensor) ** 2).mean()
            loss.backward()
            optimizer.step()

    def price(self, stock_paths, payoffs):
        """Backward induction with learned features."""
        n_paths, n_stocks, n_dates = stock_paths.shape

        continuation_value = payoffs[:, -1].copy()

        for t in range(n_dates - 1, 0, -1):
            continuation_value = np.exp(-r * dt) * continuation_value

            S_t = stock_paths[:, :, t]
            immediate = payoffs[:, t]

            itm = immediate > 0
            if itm.sum() > 0:
                # Train network on ITM paths
                self.train_network(S_t[itm], continuation_value[itm])

                # Extract features
                with torch.no_grad():
                    Phi = self.network(torch.FloatTensor(S_t[itm])).numpy()

                # Least squares in feature space
                beta = np.linalg.lstsq(Phi, continuation_value[itm], rcond=None)[0]
                predicted = Phi @ beta

                # Exercise decision
                exercise = immediate[itm] >= predicted
                continuation_value[itm] = np.where(exercise, immediate[itm], continuation_value[itm])

        continuation_value = np.exp(-r * dt) * continuation_value
        return np.mean(np.maximum(payoffs[:, 0], continuation_value))`,

    DOS: `# Deep Optimal Stopping
import torch
import torch.nn as nn
import torch.optim as optim

class DOS:
    """Deep learning approach to optimal stopping."""

    def __init__(self, hidden_size=64):
        # Stopping network: predicts stopping probability
        self.stopping_net = nn.Sequential(
            nn.Linear(n_stocks + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output probability
        )

        # Continuation network: predicts continuation value
        self.continuation_net = nn.Sequential(
            nn.Linear(n_stocks + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def train(self, stock_paths, payoffs, epochs=50):
        """Train networks using policy gradient."""
        optimizer = optim.Adam(
            list(self.stopping_net.parameters()) +
            list(self.continuation_net.parameters()),
            lr=0.001
        )

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Simulate stopping decisions
            total_payoff = 0

            for t in range(n_dates):
                # Augment state with time
                time_feature = torch.full((n_paths, 1), t / n_dates)
                state = torch.cat([
                    torch.FloatTensor(stock_paths[:, :, t]),
                    time_feature
                ], dim=1)

                # Predict stopping probability
                stop_prob = self.stopping_net(state)

                # Sample stopping decision
                stop_decision = torch.bernoulli(stop_prob)

                # Calculate payoff
                immediate = torch.FloatTensor(payoffs[:, t])
                continuation = self.continuation_net(state).squeeze()

                # Loss: maximize expected payoff
                payoff = stop_decision * immediate + (1 - stop_decision) * continuation
                total_payoff += payoff

            loss = -total_payoff.mean()
            loss.backward()
            optimizer.step()

    def price(self, stock_paths, payoffs):
        """Price option after training."""
        self.train(stock_paths, payoffs)

        # Evaluate optimal stopping strategy
        with torch.no_grad():
            values = []
            for path_idx in range(n_paths):
                for t in range(n_dates):
                    time_feature = torch.FloatTensor([[t / n_dates]])
                    state = torch.cat([
                        torch.FloatTensor(stock_paths[path_idx, :, t].reshape(1, -1)),
                        time_feature
                    ], dim=1)

                    stop_prob = self.stopping_net(state).item()

                    if stop_prob > 0.5 or t == n_dates - 1:
                        values.append(payoffs[path_idx, t])
                        break

        return np.mean(values)`,

    FQI: `# Fitted Q-Iteration
import numpy as np
from sklearn.neural_network import MLPRegressor

class FQI:
    """Q-learning approach with neural network approximation."""

    def __init__(self, hidden_size=100):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size),
            activation='tanh',
            solver='adam',
            max_iter=200,
            random_state=42
        )

    def price(self, stock_paths, payoffs):
        """Fitted Q-iteration algorithm."""
        n_paths, n_stocks, n_dates = stock_paths.shape

        # Initialize Q-values
        Q_values = payoffs[:, -1].copy()

        # Collect training data
        for t in range(n_dates - 1, 0, -1):
            # Discount Q-values
            Q_values = np.exp(-r * dt) * Q_values

            # Prepare state features
            time_features = np.full((n_paths, 1), t / n_dates)
            states = np.hstack([
                stock_paths[:, :, t].reshape(n_paths, -1),
                time_features
            ])

            # Fit Q-function approximator
            self.model.fit(states, Q_values)

            # Predict continuation values
            Q_continue = self.model.predict(states)

            # Compute immediate exercise values
            Q_exercise = payoffs[:, t]

            # Bellman update: Q = max(exercise, continue)
            Q_values = np.maximum(Q_exercise, Q_continue)

        # Evaluate at t=0
        time_0 = np.zeros((n_paths, 1))
        states_0 = np.hstack([
            stock_paths[:, :, 0].reshape(n_paths, -1),
            time_0
        ])
        Q_0 = self.model.predict(states_0)

        # Option value
        option_value = np.maximum(payoffs[:, 0], Q_0)

        return np.mean(option_value)`,
  };

  return (
    <section id="code" className="section bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4 text-gray-900">
              Algorithm Implementations
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Explore the code behind each algorithm with interactive examples
            </p>
          </div>

          {/* Algorithm selector */}
          <div className="flex flex-wrap justify-center gap-3 mb-8">
            {(['RLSM', 'RFQI', 'LSM', 'NLSM', 'DOS', 'FQI'] as Algorithm[]).map((algo) => (
              <button
                key={algo}
                onClick={() => setSelectedAlgo(algo)}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  selectedAlgo === algo
                    ? 'bg-blue-600 text-white shadow-lg scale-105'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {algo}
              </button>
            ))}
          </div>

          {/* Code editor */}
          <CodeEditor
            title={`${selectedAlgo} Implementation`}
            language="python"
            defaultCode={codeExamples[selectedAlgo]}
            height="600px"
            readOnly={true}
          />

          {/* Key insights */}
          <div className="mt-8 p-6 bg-blue-50 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Key Implementation Details
            </h3>
            <ul className="space-y-2 text-gray-700">
              {selectedAlgo === 'RLSM' && (
                <>
                  <li>• <strong>Random features:</strong> Weight matrix W is initialized randomly and never trained</li>
                  <li>• <strong>Least squares:</strong> Only output coefficients β are learned via regression</li>
                  <li>• <strong>Efficiency:</strong> No gradient descent needed - just one linear solve per timestep</li>
                  <li>• <strong>Scalability:</strong> Works well with high-dimensional problems</li>
                </>
              )}
              {selectedAlgo === 'RFQI' && (
                <>
                  <li>• <strong>Q-learning:</strong> Learns state-action value function Q(s,a)</li>
                  <li>• <strong>Bellman equation:</strong> Q(s) = max(exercise, E[Q(s')])</li>
                  <li>• <strong>State augmentation:</strong> Includes time as a feature</li>
                  <li>• <strong>Random features:</strong> Uses frozen random projections for efficiency</li>
                </>
              )}
              {selectedAlgo === 'LSM' && (
                <>
                  <li>• <strong>Classic algorithm:</strong> Original Longstaff-Schwartz (2001) method</li>
                  <li>• <strong>Polynomial basis:</strong> Uses Laguerre polynomials as features</li>
                  <li>• <strong>ITM regression:</strong> Fits continuation value only on in-the-money paths</li>
                  <li>• <strong>Industry standard:</strong> Widely used in practice for American options</li>
                </>
              )}
              {selectedAlgo === 'NLSM' && (
                <>
                  <li>• <strong>Neural features:</strong> Learns optimal basis functions via neural network</li>
                  <li>• <strong>Hybrid approach:</strong> Combines neural networks with least squares</li>
                  <li>• <strong>Adaptive:</strong> Features automatically adapt to problem structure</li>
                  <li>• <strong>Training:</strong> Network weights updated at each timestep</li>
                </>
              )}
              {selectedAlgo === 'DOS' && (
                <>
                  <li>• <strong>Deep learning:</strong> End-to-end learning of optimal policy</li>
                  <li>• <strong>Two networks:</strong> Separate networks for stopping and continuation</li>
                  <li>• <strong>Policy gradient:</strong> Trained to maximize expected payoff</li>
                  <li>• <strong>Flexibility:</strong> Can handle complex payoffs and constraints</li>
                </>
              )}
              {selectedAlgo === 'FQI' && (
                <>
                  <li>• <strong>Neural Q-learning:</strong> Uses MLP to approximate Q-function</li>
                  <li>• <strong>Fitted iteration:</strong> Updates Q-values via temporal difference</li>
                  <li>• <strong>Scikit-learn:</strong> Leverages MLPRegressor for function approximation</li>
                  <li>• <strong>Robust:</strong> Works well across different problem types</li>
                </>
              )}
            </ul>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default CodeExamples;
