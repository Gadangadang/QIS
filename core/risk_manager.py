"""
Risk management module for portfolio backtesting.

This module provides position sizing, risk limits, and portfolio constraints
to prevent over-leveraging and excessive drawdowns.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from collections import deque


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_size: float = 0.25
    max_leverage: float = 1.0
    max_drawdown_stop: float = -0.20
    position_sizing_method: str = 'equal_weight'
    fixed_fraction: float = 0.02
    kelly_fraction: float = 0.5
    volatility_target: float = 0.15
    correlation_threshold: float = 0.70
    correlation_window: int = 60
    volatility_window: int = 30


class RiskManager:
    """Manages portfolio risk through position sizing and limit checks."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.returns_history: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[pd.Timestamp] = None
        self.risk_metrics_history: List[Dict] = []
        self.violations_history: List[Dict] = []
    
    def update_returns(self, ticker: str, date: pd.Timestamp, return_value: float):
        """Update rolling returns for an asset."""
        if ticker not in self.returns_history:
            self.returns_history[ticker] = deque(maxlen=self.config.volatility_window)
        self.returns_history[ticker].append(return_value)
    
    def calculate_volatility(self, ticker: str, returns: Optional[pd.Series] = None) -> float:
        """Calculate annualized volatility for an asset."""
        if returns is not None:
            vol = returns.std() * np.sqrt(252)
            self.volatility_cache[ticker] = vol
            return vol
        
        if ticker in self.volatility_cache:
            return self.volatility_cache[ticker]
        
        if ticker in self.returns_history and len(self.returns_history[ticker]) > 5:
            returns_array = np.array(list(self.returns_history[ticker]))
            vol = np.std(returns_array) * np.sqrt(252)
            self.volatility_cache[ticker] = vol
            return vol
        
        return 0.15
    
    def update_correlations(self, returns_df: pd.DataFrame):
        """Update correlation matrix from returns dataframe."""
        if len(returns_df) >= self.config.correlation_window:
            recent = returns_df.tail(self.config.correlation_window)
            self.correlation_matrix = recent.corr()
            self.last_correlation_update = returns_df.index[-1]
        elif len(returns_df) >= 10:
            self.correlation_matrix = returns_df.corr()
            self.last_correlation_update = returns_df.index[-1]
    
    def calculate_position_size(
        self,
        ticker: str,
        signal: float,
        capital: float,
        positions: Dict[str, float],
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """Calculate appropriate position size based on method."""
        method = self.config.position_sizing_method
        
        if method == 'equal_weight':
            return self._equal_weight_sizing()
        elif method == 'fixed_fraction':
            return self._fixed_fraction_sizing()
        elif method == 'kelly':
            return self._kelly_sizing(win_rate, avg_win, avg_loss)
        elif method == 'vol_adjusted':
            if volatility is None:
                volatility = self.calculate_volatility(ticker)
            return self._vol_adjusted_sizing(volatility)
        else:
            return self._equal_weight_sizing()
    
    def _equal_weight_sizing(self) -> float:
        """Equal weight allocation."""
        return self.config.max_position_size
    
    def _fixed_fraction_sizing(self) -> float:
        """Fixed fraction of capital."""
        return min(self.config.fixed_fraction, self.config.max_position_size)
    
    def _kelly_sizing(
        self,
        win_rate: Optional[float],
        avg_win: Optional[float],
        avg_loss: Optional[float]
    ) -> float:
        """Kelly criterion position sizing."""
        if win_rate is None or avg_win is None or avg_loss is None:
            return self._fixed_fraction_sizing()
        
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self._fixed_fraction_sizing()
        
        loss_rate = 1 - win_rate
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        kelly = max(0, kelly) * self.config.kelly_fraction
        
        return min(kelly, self.config.max_position_size)
    
    def _vol_adjusted_sizing(self, volatility: float) -> float:
        """Volatility-adjusted position sizing."""
        if volatility <= 0:
            return self._fixed_fraction_sizing()
        
        size = self.config.volatility_target / volatility
        return min(size, self.config.max_position_size)
    
    def validate_trade(
        self,
        ticker: str,
        size: float,
        positions: Dict[str, float],
        portfolio_value: float,
        prices: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """Validate if a trade violates any risk limits."""
        if abs(size) > self.config.max_position_size:
            reason = f"Position size {abs(size):.2%} exceeds max {self.config.max_position_size:.2%}"
            self._log_violation(ticker, "position_size", reason)
            return False, reason
        
        if prices is not None:
            total_exposure = sum(
                abs(positions.get(t, 0) * prices.get(t, 0))
                for t in positions.keys()
            )
            if ticker in prices:
                proposed_value = abs(size * portfolio_value)
                total_exposure += proposed_value
            
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            if leverage > self.config.max_leverage:
                reason = f"Leverage {leverage:.2f}x exceeds max {self.config.max_leverage:.2f}x"
                self._log_violation(ticker, "leverage", reason)
                return False, reason
        
        if self.correlation_matrix is not None and len(positions) > 0:
            correlation_risk = self._check_correlation_exposure(ticker, positions)
            if correlation_risk > self.config.correlation_threshold:
                reason = f"Correlation exposure {correlation_risk:.2%} exceeds threshold {self.config.correlation_threshold:.2%}"
                self._log_violation(ticker, "correlation", reason)
        
        return True, "OK"
    
    def _check_correlation_exposure(self, ticker: str, positions: Dict[str, float]) -> float:
        """Check correlation exposure between new ticker and existing positions."""
        if self.correlation_matrix is None:
            return 0.0
        
        if ticker not in self.correlation_matrix.index:
            return 0.0
        
        corrs = []
        for pos_ticker in positions.keys():
            if pos_ticker in self.correlation_matrix.columns and pos_ticker != ticker:
                corr = self.correlation_matrix.loc[ticker, pos_ticker]
                if not pd.isna(corr):
                    corrs.append(abs(corr))
        
        return max(corrs) if corrs else 0.0
    
    def check_stop_conditions(
        self,
        current_drawdown: float,
        equity_curve: pd.Series
    ) -> Tuple[bool, str]:
        """Check if any stop conditions are triggered."""
        if current_drawdown <= self.config.max_drawdown_stop:
            reason = f"Max drawdown {current_drawdown:.2%} exceeds stop {self.config.max_drawdown_stop:.2%}"
            self._log_violation("PORTFOLIO", "drawdown_stop", reason)
            return True, reason
        
        return False, ""
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        metrics = {}
        
        total_exposure = sum(
            abs(positions.get(t, 0) * prices.get(t, 0))
            for t in positions.keys()
        )
        metrics['leverage'] = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        weights = {}
        for ticker in positions.keys():
            if positions[ticker] != 0 and ticker in prices:
                value = positions[ticker] * prices[ticker]
                weights[ticker] = value / portfolio_value
        
        metrics['num_positions'] = len([w for w in weights.values() if w != 0])
        metrics['max_position_weight'] = max(abs(w) for w in weights.values()) if weights else 0
        
        if self.correlation_matrix is not None and len(weights) > 0:
            port_vol = self._calculate_portfolio_volatility(weights)
            metrics['portfolio_volatility'] = port_vol
        else:
            metrics['portfolio_volatility'] = 0.0
        
        return metrics
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility using correlation matrix."""
        if self.correlation_matrix is None:
            return 0.0
        
        tickers = list(weights.keys())
        if len(tickers) < 2:
            if tickers and tickers[0] in self.volatility_cache:
                return self.volatility_cache[tickers[0]]
            return 0.0
        
        vol_dict = {t: self.calculate_volatility(t) for t in tickers}
        
        portfolio_var = 0.0
        for t1 in tickers:
            for t2 in tickers:
                if t1 in self.correlation_matrix.index and t2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[t1, t2]
                    if not pd.isna(corr):
                        cov = corr * vol_dict[t1] * vol_dict[t2]
                        portfolio_var += weights[t1] * weights[t2] * cov
        
        return np.sqrt(max(0, portfolio_var))
    
    def _log_violation(self, ticker: str, violation_type: str, reason: str):
        """Log a risk violation for tracking."""
        self.violations_history.append({
            'ticker': ticker,
            'type': violation_type,
            'reason': reason,
            'timestamp': pd.Timestamp.now()
        })
    
    def log_metrics(
        self,
        date: pd.Timestamp,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        drawdown: float
    ):
        """Log risk metrics for visualization."""
        risk_metrics = self.calculate_portfolio_risk(positions, prices, portfolio_value)
        risk_metrics['date'] = date
        risk_metrics['drawdown'] = drawdown
        self.risk_metrics_history.append(risk_metrics)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get risk metrics history as DataFrame."""
        if not self.risk_metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.risk_metrics_history)
    
    def get_violations_dataframe(self) -> pd.DataFrame:
        """Get violations history as DataFrame."""
        if not self.violations_history:
            return pd.DataFrame()
        return pd.DataFrame(self.violations_history)
