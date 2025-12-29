"""
TAA Portfolio Optimizer using Mean-Variance Optimization.
Uses cvxpy for convex optimization with constraints.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple
import logging

from .constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class TAAOptimizer:
    """
    Mean-variance portfolio optimizer for TAA strategy.
    
    Objective:
        maximize: expected_return - risk_aversion * portfolio_variance - transaction_costs
    
    Subject to:
        - weights sum to 1 (fully invested or with cash)
        - position size limits (min/max per sector)
        - tracking error constraint (vs benchmark)
        - turnover constraint (vs previous weights)
    
    Example:
        >>> optimizer = TAAOptimizer(constraints)
        >>> weights = optimizer.optimize(
        ...     expected_returns={'XLK': 0.08, 'XLF': 0.05},
        ...     covariance_matrix=cov_matrix,
        ...     previous_weights={'XLK': 0.3, 'XLF': 0.2}
        ... )
    """
    
    def __init__(self, constraints: OptimizationConstraints):
        """
        Initialize optimizer with constraints.
        
        Args:
            constraints: OptimizationConstraints object with all parameters
        """
        self.constraints = constraints
        self.constraints.validate_all()
    
    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        previous_weights: Optional[Dict[str, float]] = None,
        current_vix: Optional[float] = None
    ) -> Tuple[Dict[str, float], Dict[str, any]]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Dict of ticker -> expected return (e.g., 0.08 for 8%)
            covariance_matrix: Covariance matrix of returns (tickers as index/columns)
            previous_weights: Previous portfolio weights (for turnover calculation)
            current_vix: Current VIX level (for transaction cost adjustment)
            
        Returns:
            Tuple of:
                - Dict[str, float]: Optimal weights (ticker -> weight)
                - Dict[str, any]: Optimization metadata (objective value, status, etc.)
        """
        # Convert inputs to arrays with consistent ordering
        tickers = list(expected_returns.keys())
        n_assets = len(tickers)
        
        mu = np.array([expected_returns[t] for t in tickers])
        Sigma = covariance_matrix.loc[tickers, tickers].values
        
        # Previous weights (default to equal weight if not provided)
        if previous_weights is None:
            w_prev = np.ones(n_assets) / n_assets
        else:
            w_prev = np.array([previous_weights.get(t, 0) for t in tickers])
        
        # Define optimization variable
        w = cp.Variable(n_assets)
        
        # Objective: maximize expected return - risk penalty - transaction costs - turnover penalty
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        
        # Turnover (sum of absolute changes)
        turnover = cp.sum(cp.abs(w - w_prev))
        
        # Transaction costs
        tc_model = self.constraints.transaction_costs
        tc_rate = (tc_model.commission_bps + tc_model.slippage_bps) / 10000
        if tc_model.vix_adjustment_enabled and current_vix and current_vix > tc_model.vix_threshold:
            tc_rate *= tc_model.vix_multiplier
        transaction_costs = tc_rate * turnover
        
        # Turnover penalty
        turnover_penalty = 0
        if self.constraints.turnover.penalize:
            turnover_penalty = self.constraints.turnover.penalty_lambda * turnover
        
        # Objective function
        objective = cp.Maximize(
            portfolio_return 
            - self.constraints.risk_aversion * portfolio_variance
            - transaction_costs
            - turnover_penalty
        )
        
        # Constraints list
        constraints_list = []
        
        # 1. Weights sum to 1
        constraints_list.append(cp.sum(w) == 1)
        
        # 2. Position limits
        pos_constr = self.constraints.position
        if pos_constr.long_only:
            constraints_list.append(w >= pos_constr.min_position)
        else:
            constraints_list.append(w >= -pos_constr.max_position)  # Allow shorts
        
        constraints_list.append(w <= pos_constr.max_position)
        
        # 3. Tracking error constraint
        if self.constraints.tracking_error.enabled:
            te_constr = self.constraints.tracking_error
            w_bench = np.array([te_constr.benchmark_weights.get(t, 0) for t in tickers])
            
            # Active weights
            w_active = w - w_bench
            
            # Tracking error: sqrt((w - w_bench)' * Sigma * (w - w_bench))
            # Constraint: TE <= max_te
            # Squared form: (w - w_bench)' * Sigma * (w - w_bench) <= max_te^2
            te_squared = cp.quad_form(w_active, Sigma)
            constraints_list.append(te_squared <= te_constr.max_te ** 2)
        
        # 4. Turnover constraint
        if self.constraints.turnover.enabled:
            max_turnover = self.constraints.turnover.max_monthly
            constraints_list.append(turnover <= max_turnover)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        # Try solvers in order of preference: CLARABEL > OSQP > SCS
        # ECOS not available, using installed solvers
        solvers_to_try = [cp.CLARABEL, cp.OSQP, cp.SCS]
        solver_used = None
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                solver_used = solver
                break
            except Exception as e:
                logger.debug(f"Solver {solver} failed: {e}")
                continue
        
        if solver_used is None:
            logger.error("All solvers failed")
            # Fallback to equal weight
            weights_dict = {t: 1/n_assets for t in tickers}
            metadata = {
                'status': 'failed',
                'solver_status': 'All solvers failed',
                'objective_value': None,
                'expected_return': None,
                'volatility': None,
                'turnover': None
            }
            return weights_dict, metadata
        
        # Extract optimal weights
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.warning(f"Optimization status: {problem.status} (solver: {solver_used})")
            # Fallback to equal weight
            weights_dict = {t: 1/n_assets for t in tickers}
            metadata = {
                'status': problem.status,
                'solver_status': f"{solver_used}: {problem.status}",
                'objective_value': None,
                'expected_return': None,
                'volatility': None,
                'turnover': None
            }
            return weights_dict, metadata
        
        # Successful optimization
        optimal_weights = w.value
        weights_dict = {t: max(0, w_val) for t, w_val in zip(tickers, optimal_weights)}
        
        # Renormalize to ensure sum=1 (numerical precision)
        total = sum(weights_dict.values())
        weights_dict = {t: w/total for t, w in weights_dict.items()}
        
        # Calculate realized metrics
        realized_return = mu @ optimal_weights
        realized_variance = optimal_weights @ Sigma @ optimal_weights
        realized_volatility = np.sqrt(realized_variance) * np.sqrt(252)  # Annualize
        realized_turnover = np.sum(np.abs(optimal_weights - w_prev))
        
        metadata = {
            'status': 'optimal',
            'solver_status': f"{solver_used}: {problem.status}",
            'objective_value': problem.value,
            'expected_return': realized_return,
            'volatility': realized_volatility,
            'turnover': realized_turnover,
            'tracking_error': self._calculate_tracking_error(optimal_weights, tickers, Sigma) if self.constraints.tracking_error.enabled else None
        }
        
        logger.info(f"Optimization successful (solver: {solver_used}): Return={realized_return:.4f}, Vol={realized_volatility:.4f}, Turnover={realized_turnover:.4f}")
        
        return weights_dict, metadata
    
    def _calculate_tracking_error(
        self, 
        weights: np.ndarray, 
        tickers: list, 
        covariance_matrix: np.ndarray
    ) -> float:
        """
        Calculate tracking error vs benchmark.
        
        Args:
            weights: Portfolio weights
            tickers: List of tickers
            covariance_matrix: Covariance matrix
            
        Returns:
            float: Annualized tracking error
        """
        w_bench = np.array([
            self.constraints.tracking_error.benchmark_weights.get(t, 0) 
            for t in tickers
        ])
        w_active = weights - w_bench
        te_variance = w_active @ covariance_matrix @ w_active
        return np.sqrt(te_variance) * np.sqrt(252)  # Annualize


class BacktestOptimizer:
    """
    Wrapper for running optimizer over historical backtest.
    Handles rolling window optimization with lookback periods.
    """
    
    def __init__(
        self, 
        optimizer: TAAOptimizer,
        lookback_days: int = 252,  # 1 year of data for cov estimation
        rebalance_freq: str = 'W'  # Weekly rebalancing
    ):
        """
        Initialize backtest optimizer.
        
        Args:
            optimizer: TAAOptimizer instance
            lookback_days: Days of history for covariance estimation
            rebalance_freq: Rebalance frequency ('D', 'W', 'M')
        """
        self.optimizer = optimizer
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
    
    def run_backtest(
        self,
        predictions_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Run backtest with periodic rebalancing.
        
        Args:
            predictions_df: DataFrame with predicted returns (Date index, ticker columns)
            returns_df: Historical returns for covariance estimation (Date index, ticker columns)
            vix_series: VIX time series for transaction cost adjustment
            
        Returns:
            pd.DataFrame: Portfolio weights over time with metadata columns
        """
        logger.info(f"Starting backtest: {len(predictions_df)} prediction dates, {len(returns_df)} return dates")
        logger.info(f"Predictions range: {predictions_df.index.min()} to {predictions_df.index.max()}")
        logger.info(f"Returns range: {returns_df.index.min()} to {returns_df.index.max()}")
        
        # Resample to rebalance frequency
        # Resample gets last available prediction in each period
        rebalanced_predictions = predictions_df.resample(self.rebalance_freq).last()
        # Drop any NaN rows (periods with no data)
        rebalanced_predictions = rebalanced_predictions.dropna(how='all')
        logger.info(f"Rebalance dates ({self.rebalance_freq}): {len(rebalanced_predictions)} dates")
        
        results = []
        previous_weights = None
        skipped_count = 0
        
        for i, date in enumerate(rebalanced_predictions.index):
            try:
                if i == 0:
                    logger.info(f"Processing first rebalance date: {date}")
                
                # Get predictions from resampled data
                expected_returns = rebalanced_predictions.loc[date].to_dict()
                
                # Calculate covariance matrix using lookback window
                lookback_start = date - pd.Timedelta(days=self.lookback_days)
                historical_returns = returns_df.loc[lookback_start:date]
                
                if len(historical_returns) < 20:  # Minimum data required
                    logger.warning(f"Insufficient data at {date} ({len(historical_returns)} days), skipping")
                    continue
                
                cov_matrix = historical_returns.cov() * 252  # Annualize
                
                # Get VIX if available
                current_vix = vix_series.loc[date] if vix_series is not None and date in vix_series.index else None
                
                # Optimize
                weights, metadata = self.optimizer.optimize(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    previous_weights=previous_weights,
                    current_vix=current_vix
                )
                
                # Store results
                result = {'Date': date}
                result.update(weights)
                result.update({f'meta_{k}': v for k, v in metadata.items()})
                results.append(result)
                
                previous_weights = weights
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(rebalance_dates)} rebalance dates, {len(results)} successful")
            
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}", exc_info=True)
                continue
        
        logger.info(f"Backtest complete: {len(results)} results, {skipped_count} skipped")
        
        if not results:
            raise RuntimeError(
                f"No valid optimization results generated. "
                f"Check that predictions and returns date ranges overlap. "
                f"Predictions: {predictions_df.index.min()} to {predictions_df.index.max()}, "
                f"Returns: {returns_df.index.min()} to {returns_df.index.max()}"
            )
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results).set_index('Date')
        logger.info(f"Backtest complete: {len(results_df)} portfolio weights generated")
        
        return results_df
