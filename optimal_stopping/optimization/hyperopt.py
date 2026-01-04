"""
Main hyperparameter optimization orchestrator using Optuna.

Supports:
- Bayesian Optimization (TPE)
- Random Search baseline
- Multi-fidelity optimization
- Automatic visualization and logging
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler
import numpy as np
import os
import json
import datetime
import subprocess
from pathlib import Path

from .search_spaces import get_search_space, suggest_hyperparameter
from .objective import evaluate_objective, evaluate_objective_with_early_stopping


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna.

    Handles the complete optimization workflow:
    1. Set up Optuna study with chosen sampler (TPE or Random)
    2. Run optimization trials
    3. Log results and metadata
    4. Generate visualizations
    5. Return best hyperparameters

    Args:
        algo_name: Algorithm name ('RLSM', 'SRLSM', 'RFQI', etc.)
        algo_class: Algorithm class
        model_class: Stock model class
        problem_config: Problem specification dictionary
        method: Optimization method ('tpe' or 'random')
        timeout: Timeout in seconds (None = no timeout)
        n_trials: Number of trials (None = run until timeout)
        variance_penalty: Weight for variance penalty in objective
        fidelity_factor: Reduction factor for nb_paths during optimization
        output_dir: Directory for saving results
        study_name: Optional custom study name
    """

    def __init__(self, algo_name, algo_class, model_class, problem_config,
                 method='tpe', timeout=1200, n_trials=None,
                 variance_penalty=0.1, fidelity_factor=4,
                 output_dir='hyperopt_results', study_name=None):

        self.algo_name = algo_name
        self.algo_class = algo_class
        self.model_class = model_class
        self.problem_config = problem_config
        self.method = method.lower()
        self.timeout = timeout
        self.n_trials = n_trials
        self.variance_penalty = variance_penalty
        self.fidelity_factor = fidelity_factor

        # Output directory setup
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Study name
        if study_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study_name = f"{algo_name}_{method}_{timestamp}"
        else:
            self.study_name = study_name

        # Get search space for this algorithm
        self.search_space = get_search_space(algo_name)

        # Study and results
        self.study = None
        self.best_params = None
        self.best_value = None

    def optimize(self):
        """
        Run hyperparameter optimization.

        Returns:
            dict: Best hyperparameters found
        """
        print(f"\n{'='*80}")
        print(f"Starting Hyperparameter Optimization: {self.study_name}")
        print(f"{'='*80}")
        print(f"Algorithm: {self.algo_name}")
        print(f"Method: {self.method.upper()}")
        print(f"Timeout: {self.timeout}s" if self.timeout else f"Trials: {self.n_trials}")
        print(f"Multi-fidelity: Using 1/{self.fidelity_factor} of full paths")
        print(f"Search space: {self.search_space}")
        print(f"{'='*80}\n")

        # Create sampler based on method
        if self.method == 'tpe':
            sampler = TPESampler(seed=42)
        elif self.method == 'random':
            sampler = RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'tpe' or 'random'")

        # Create Optuna study
        # Use SQLite database for persistence
        storage_path = self.output_dir / f"{self.study_name}.db"
        storage = f"sqlite:///{storage_path}"

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            sampler=sampler,
            direction='maximize',  # Maximize validation price
            load_if_exists=True
        )

        # Define objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters from search space
            hyperparams = {}
            for param_name, param_spec in self.search_space.items():
                hyperparams[param_name] = suggest_hyperparameter(
                    trial, param_name, param_spec
                )

            # Add early stopping for RFQI/SRFQI
            early_stopping_callback = None
            if self.algo_name in ['RFQI', 'SRFQI']:
                from .early_stopping import EarlyStopping
                # Set divergence threshold: 5x spot price (realistic upper bound for option)
                spot = self.problem_config['model_params']['spot']
                divergence_threshold = 5.0 * spot

                hyperparams['nb_epochs'] = 100  # Max epochs, early stopping will cut this short
                early_stopping_callback = EarlyStopping(
                    patience=5,       # Stop if no improvement for 5 epochs
                    min_delta=0.01,   # 1 cent improvement threshold
                    mode='max',       # Higher validation score is better
                    divergence_threshold=divergence_threshold  # Detect explosions
                )
                hyperparams['early_stopping_callback'] = early_stopping_callback

            # Evaluate objective
            obj_value, metrics = evaluate_objective(
                self.algo_class,
                self.model_class,
                hyperparams,
                self.problem_config,
                variance_penalty=self.variance_penalty,
                n_runs=3,  # Single run per trial for speed
                fidelity_factor=self.fidelity_factor
            )

            # Check for divergence
            if early_stopping_callback is not None and early_stopping_callback.diverged:
                print(f"  ⚠️  Trial {trial.number} DIVERGED - Penalizing objective")
                trial.set_user_attr('diverged', True)
                return -1e10  # Very low score to make Optuna avoid this region

            # Check for NaN/Inf in metrics
            if np.isnan(obj_value) or np.isinf(obj_value):
                print(f"  ⚠️  Trial {trial.number} returned NaN/Inf - Penalizing objective")
                trial.set_user_attr('diverged', True)
                return -1e10

            # Log intermediate metrics (convert numpy types to Python types for JSON serialization)
            trial.set_user_attr('mean_price', float(metrics['mean_price']))
            trial.set_user_attr('std_price', float(metrics['std_price']))
            trial.set_user_attr('mean_time', float(metrics['mean_time']))
            trial.set_user_attr('nb_paths_used', int(metrics['nb_paths_used']))
            trial.set_user_attr('diverged', False)

            # Log epochs used for RFQI/SRFQI
            if 'nb_epochs_used' in metrics:
                trial.set_user_attr('nb_epochs_used', int(metrics['nb_epochs_used']))

            return obj_value

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Extract best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        print(f"\n{'='*80}")
        print(f"Optimization Complete!")
        print(f"{'='*80}")
        print(f"Best hyperparameters: {self.best_params}")
        print(f"Best objective value: {self.best_value:.6f}")

        # Print nb_epochs_used for RFQI/SRFQI
        if hasattr(self.study.best_trial, 'user_attrs') and 'nb_epochs_used' in self.study.best_trial.user_attrs:
            nb_epochs = self.study.best_trial.user_attrs['nb_epochs_used']
            print(f"Epochs used (early stopping): {nb_epochs}")
            print(f"  → Use nb_epochs={nb_epochs} for final runs")

        print(f"Number of trials completed: {len(self.study.trials)}")
        print(f"{'='*80}\n")

        # Save results
        self._save_results()

        # Generate visualizations
        self._generate_visualizations()

        return self.best_params

    def _save_results(self):
        """Save optimization results and metadata."""
        # Get git commit hash for reproducibility
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
        except:
            git_hash = 'unknown'

        # Prepare metadata
        metadata = {
            'experiment_id': self.study_name,
            'date': datetime.datetime.now().isoformat(),
            'git_commit_hash': git_hash,
            'algorithm': self.algo_name,
            'optimization_method': self.method,
            'timeout_seconds': self.timeout,
            'n_trials_requested': self.n_trials,
            'n_trials_completed': len(self.study.trials),
            'variance_penalty': self.variance_penalty,
            'fidelity_factor': self.fidelity_factor,
            'problem_config': {
                'nb_paths_full': self.problem_config.get('nb_paths_full'),
                'nb_dates': self.problem_config.get('nb_dates'),
                'maturity': self.problem_config.get('maturity'),
                'model_name': self.model_class.__name__,
                'payoff_name': self.problem_config['payoff'].__class__.__name__,
            },
            'best_hyperparameters': self.best_params,
            'best_objective_value': self.best_value,
        }

        # Add nb_epochs_used for RFQI/SRFQI if available
        if hasattr(self.study.best_trial, 'user_attrs') and 'nb_epochs_used' in self.study.best_trial.user_attrs:
            metadata['nb_epochs_used'] = self.study.best_trial.user_attrs['nb_epochs_used']

        # Save as JSON
        json_path = self.output_dir / f"{self.study_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save human-readable summary
        summary_path = self.output_dir / f"{self.study_name}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Experiment ID: {metadata['experiment_id']}\n")
            f.write(f"Date: {metadata['date']}\n")
            f.write(f"Git Commit Hash: {metadata['git_commit_hash']}\n\n")

            f.write("PROBLEM SPECIFICATION:\n")
            f.write("-"*80 + "\n")
            f.write(f"Algorithm: {self.algo_name}\n")
            f.write(f"Stock Model: {metadata['problem_config']['model_name']}\n")
            f.write(f"Payoff: {metadata['problem_config']['payoff_name']}\n")
            f.write(f"Paths (full): {metadata['problem_config']['nb_paths_full']}\n")
            f.write(f"Dates: {metadata['problem_config']['nb_dates']}\n")
            f.write(f"Maturity: {metadata['problem_config']['maturity']}\n\n")

            f.write("OPTIMIZATION SETTINGS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Method: {metadata['optimization_method'].upper()}\n")
            f.write(f"Timeout: {metadata['timeout_seconds']}s\n")
            f.write(f"Trials Completed: {metadata['n_trials_completed']}\n")
            f.write(f"Variance Penalty: {metadata['variance_penalty']}\n")
            f.write(f"Multi-Fidelity Factor: 1/{metadata['fidelity_factor']}\n\n")

            f.write("BEST RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Objective Value: {metadata['best_objective_value']:.6f}\n")
            f.write("Best Hyperparameters:\n")
            for param, value in metadata['best_hyperparameters'].items():
                f.write(f"  {param}: {value}\n")

            # Add nb_epochs_used for RFQI/SRFQI
            if 'nb_epochs_used' in metadata:
                f.write(f"\nEpochs Used (with early stopping): {metadata['nb_epochs_used']}\n")
                f.write(f"  → Use nb_epochs={metadata['nb_epochs_used']} for final runs\n")

            f.write("\n" + "="*80 + "\n")

        print(f"Results saved to: {self.output_dir}")
        print(f"  - SQLite database: {self.study_name}.db")
        print(f"  - JSON summary: {self.study_name}_summary.json")
        print(f"  - Text summary: {self.study_name}_summary.txt")

    def _generate_visualizations(self):
        """Generate Optuna visualization plots."""
        try:
            import optuna.visualization as vis
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            print(f"\nGenerating visualizations...")

            # 1. Optimization history
            fig1 = vis.plot_optimization_history(self.study)
            fig1.write_image(str(self.output_dir / f"{self.study_name}_history.png"))

            # 2. Parameter importances
            try:
                fig2 = vis.plot_param_importances(self.study)
                fig2.write_image(str(self.output_dir / f"{self.study_name}_importances.png"))
            except:
                print("  Warning: Could not generate parameter importance plot (requires multiple trials)")

            # 3. Slice plot (shows where good parameters cluster)
            try:
                fig3 = vis.plot_slice(self.study)
                fig3.write_image(str(self.output_dir / f"{self.study_name}_slice.png"))
            except:
                print("  Warning: Could not generate slice plot")

            # 4. Parallel coordinate plot
            try:
                fig4 = vis.plot_parallel_coordinate(self.study)
                fig4.write_image(str(self.output_dir / f"{self.study_name}_parallel.png"))
            except:
                print("  Warning: Could not generate parallel coordinate plot")

            print(f"Visualizations saved to: {self.output_dir}")
            print(f"  - Optimization history: {self.study_name}_history.png")
            print(f"  - Parameter importances: {self.study_name}_importances.png")
            print(f"  - Slice plot: {self.study_name}_slice.png")

        except ImportError as e:
            print(f"\nWarning: Could not generate visualizations. Missing dependency: {e}")
            print("Install with: pip install plotly kaleido")
        except Exception as e:
            print(f"\nWarning: Visualization failed with error: {e}")

    def get_best_params(self):
        """Return best hyperparameters found."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        return self.best_params

    def get_study(self):
        """Return Optuna study object for advanced analysis."""
        return self.study
