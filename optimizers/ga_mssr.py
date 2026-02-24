"""
GA-MSSR optimizer: Genetic Algorithm Maximizing Sharpe-Sterling Ratio.

Uses PyGAD to optimize a 16-dimensional weight vector for combining
the Khushi trading rule signals. Each gene is a weight in [-1, 1]
determining how much to trust each rule's signal.

Portfolio position at each bar: position = weights @ signals.T
Portfolio return at each bar: port_return = position * logr
Fitness: SSR(port_returns) with max drawdown constraint.
"""
import numpy as np
import pandas as pd
import pygad
from dataclasses import dataclass, field
from typing import Optional

from optimizers.ssr import SSR, SSRResult


@dataclass
class GAMSSRConfig:
    """Configuration for the GA-MSSR optimizer."""
    sol_per_pop: int = 20
    num_parents_mating: int = 10
    num_generations: int = 200
    num_genes: int = 16
    gene_low: float = -1.0
    gene_high: float = 1.0
    max_drawdown_pct: float = 0.15
    mutation_probability: float = 0.1
    crossover_type: str = "single_point"
    parent_selection_type: str = "sss"
    random_seed: Optional[int] = None


@dataclass
class GAMSSRResult:
    """Result of a GA-MSSR optimization run."""
    best_weights: np.ndarray
    best_fitness: float
    fitness_history: list[float]
    ssr_result: SSRResult
    num_generations_run: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    oos_returns: list[pd.Series]
    oos_ssr: list[float]
    aggregate_ssr: float
    total_oos_return: float
    num_folds: int


class GAMSSR:
    """
    Genetic Algorithm optimizer for the Khushi 16-rule trading system.

    Optimizes a weight vector w ∈ [-1, 1]^16 to maximize the
    Sharpe-Sterling Ratio of the weighted signal combination.

    Parameters
    ----------
    config : GAMSSRConfig
        GA hyperparameters and constraints.
    """

    def __init__(self, config: Optional[GAMSSRConfig] = None):
        self.config = config or GAMSSRConfig()
        self._ssr = SSR()
        self._signals: Optional[np.ndarray] = None
        self._logr: Optional[np.ndarray] = None
        self._best_weights: Optional[np.ndarray] = None
        self._fitness_history: list[float] = []

    def fit(self, features_df: pd.DataFrame) -> GAMSSRResult:
        """
        Run GA optimization on a feature matrix.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of get_rule_features(): columns logr, rule_1..rule_16.

        Returns
        -------
        GAMSSRResult
        """
        # Extract numpy arrays
        self._logr = features_df["logr"].values.astype(np.float64)
        signal_cols = [c for c in features_df.columns if c.startswith("rule_")]
        self._signals = features_df[signal_cols].values.astype(np.float64)
        self._fitness_history = []

        num_genes = self._signals.shape[1]

        def fitness_func(ga_instance, solution, solution_idx):
            return self._evaluate(solution)

        def on_generation(ga_instance):
            best = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
            self._fitness_history.append(float(best[1]))

        ga_instance = pygad.GA(
            num_generations=self.config.num_generations,
            num_parents_mating=self.config.num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=self.config.sol_per_pop,
            num_genes=num_genes,
            gene_type=float,
            init_range_low=self.config.gene_low,
            init_range_high=self.config.gene_high,
            gene_space={"low": self.config.gene_low, "high": self.config.gene_high},
            mutation_probability=self.config.mutation_probability,
            crossover_type=self.config.crossover_type,
            parent_selection_type=self.config.parent_selection_type,
            on_generation=on_generation,
            random_seed=self.config.random_seed,
            suppress_warnings=True,
        )

        ga_instance.run()

        # Extract best solution
        best_solution, best_fitness, _ = ga_instance.best_solution()
        self._best_weights = np.array(best_solution, dtype=np.float64)

        # Compute detailed SSR for the best solution
        port_returns = self._compute_returns(self._best_weights)
        ssr_result = self._ssr.calculate(port_returns)

        return GAMSSRResult(
            best_weights=self._best_weights,
            best_fitness=float(best_fitness),
            fitness_history=self._fitness_history,
            ssr_result=ssr_result,
            num_generations_run=self.config.num_generations,
        )

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions and returns using trained weights.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix (logr + rule signals).

        Returns
        -------
        pd.DataFrame
            Columns: position, port_return, cum_return.
        """
        if self._best_weights is None:
            raise RuntimeError("Must call fit() before predict()")

        logr = features_df["logr"].values.astype(np.float64)
        signal_cols = [c for c in features_df.columns if c.startswith("rule_")]
        signals = features_df[signal_cols].values.astype(np.float64)

        position = signals @ self._best_weights
        port_return = position * logr
        cum_return = np.cumsum(port_return)

        return pd.DataFrame(
            {
                "position": position,
                "port_return": port_return,
                "cum_return": cum_return,
            },
            index=features_df.index,
        )

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
        periods: Optional[list[int]] = None,
    ) -> WalkForwardResult:
        """
        Walk-forward analysis: rolling train/test optimization.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV DataFrame (with lowercase columns).
        train_size : int
            Number of bars in each training window.
        test_size : int
            Number of bars in each test window.
        step_size : int, optional
            Step between windows. Defaults to test_size (non-overlapping).
        periods : list[int], optional
            Period grid for rule param training. Defaults to small grid for speed.
        """
        from strategies.khushi_rules import train_rule_params, get_rule_features

        if step_size is None:
            step_size = test_size
        if periods is None:
            periods = [3, 7, 15, 27, 50]

        n = len(df)
        oos_returns_list = []
        oos_ssr_list = []

        start = 0
        while start + train_size + test_size <= n:
            train_df = df.iloc[start : start + train_size]
            test_df = df.iloc[start + train_size : start + train_size + test_size]

            # Train rule params on training window
            rule_params = train_rule_params(train_df, periods=periods)

            # Get features for both windows
            train_features = get_rule_features(train_df, rule_params)
            test_features = get_rule_features(test_df, rule_params)

            if len(train_features) < 10 or len(test_features) < 5:
                start += step_size
                continue

            # Fit GA on training data
            result = self.fit(train_features)

            # Evaluate on test data
            test_pred = self.predict(test_features)
            oos_ret = test_pred["port_return"]
            oos_returns_list.append(oos_ret)

            oos_result = self._ssr.calculate(oos_ret.values)
            oos_ssr_list.append(oos_result.ssr)

            start += step_size

        # Aggregate OOS results
        if oos_returns_list:
            all_oos = pd.concat(oos_returns_list)
            agg_result = self._ssr.calculate(all_oos.values)
            aggregate_ssr = agg_result.ssr
            total_return = float(all_oos.sum())
        else:
            aggregate_ssr = 0.0
            total_return = 0.0

        return WalkForwardResult(
            oos_returns=oos_returns_list,
            oos_ssr=oos_ssr_list,
            aggregate_ssr=aggregate_ssr,
            total_oos_return=total_return,
            num_folds=len(oos_returns_list),
        )

    def _evaluate(self, solution: np.ndarray) -> float:
        """Evaluate a single chromosome's fitness with drawdown constraint."""
        port_returns = self._compute_returns(solution)

        # Max drawdown constraint
        cumulative = np.cumsum(port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        peak = np.max(np.abs(running_max))
        if peak > 1e-10:
            max_dd_pct = abs(np.min(drawdown)) / peak
            if max_dd_pct > self.config.max_drawdown_pct:
                return -1e9

        result = self._ssr.calculate(port_returns)
        return result.ssr

    def _compute_returns(self, weights: np.ndarray) -> np.ndarray:
        """Compute portfolio returns for a weight vector."""
        position = self._signals @ weights
        return position * self._logr
