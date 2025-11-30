"""
RiskManager class - Enforces risk rules and monitors portfolio risk.

Responsibilities:
- Check risk limits (stops, concentration)
- Monitor portfolio-level risk
- Kill switches and circuit breakers for catastrophic losses
- Position sizing delegated to PositionSizer classes

Note: Position sizing logic has been moved to position_sizers.py for better separation of concerns.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime
import numpy as np
import pandas as pd

from .position_sizers import PositionSizer, FixedFractionalSizer


@dataclass
class RiskConfig:
    """Risk management configuration."""
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.20  # Max 20% in one position
    max_portfolio_leverage: float = 1.0  # No leverage by default
    stop_loss_pct: Optional[float] = None  # Stop loss percentage (e.g., 0.10 = 10%)
    take_profit_pct: Optional[float] = None  # Take profit percentage
    max_correlation_exposure: float = 0.50  # Max 50% in correlated assets
    min_trade_value: float = 100.0  # Minimum trade size
    
    # Kill switches and circuit breakers
    max_drawdown_pct: float = 0.15  # Max 15% drawdown kill switch
    max_daily_loss_pct: float = 0.03  # Max 3% daily loss kill switch
    max_portfolio_heat_pct: float = 0.10  # Max 10% total portfolio at risk
    min_capital_pct: float = 0.50  # Kill switch if capital falls below 50% of peak
    

class RiskManager:
    """
    Enforces risk rules and monitors portfolio risk.
    
    Position sizing is now handled by injected PositionSizer classes.
    
    Example:
        from core.portfolio.position_sizers import FixedFractionalSizer
        
        config = RiskConfig(risk_per_trade=0.02, max_position_size=0.20, stop_loss_pct=0.10)
        sizer = FixedFractionalSizer(max_position_pct=0.20, risk_per_trade=0.02)
        risk_mgr = RiskManager(config, position_sizer=sizer)
        
        shares = risk_mgr.calculate_position_size(
            ticker='ES',
            signal=1.0,
            current_price=4500.0,
            portfolio_value=100000
        )
        
        should_exit = risk_mgr.check_stop_loss(position)
    """
    
    def __init__(self, config: RiskConfig, position_sizer: Optional[PositionSizer] = None):
        """
        Initialize risk manager with configuration.
        
        Args:
            config: RiskConfig object with risk parameters
            position_sizer: PositionSizer instance (if None, uses FixedFractionalSizer)
        """
        self.config = config
        
        # Use provided sizer or create default
        if position_sizer is None:
            self.position_sizer = FixedFractionalSizer(
                max_position_pct=config.max_position_size,
                risk_per_trade=config.risk_per_trade,
                min_trade_value=config.min_trade_value
            )
        else:
            self.position_sizer = position_sizer
    
    def calculate_position_size(
        self, 
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate number of shares to buy using the configured position sizer.
        
        This method now delegates to the injected PositionSizer instance.
        
        Args:
            ticker: Asset ticker
            signal: Signal strength (-1 to 1, where 1 = full long, -1 = full short)
            current_price: Current price per share
            portfolio_value: Current portfolio value
            volatility: Optional volatility measure (annualized std)
            **kwargs: Additional parameters for specific sizers (atr, win_rate, etc.)
            
        Returns:
            Number of shares to buy (integer)
        """
        # Delegate to position sizer
        return self.position_sizer.calculate_size(
            ticker=ticker,
            signal=signal,
            current_price=current_price,
            portfolio_value=portfolio_value,
            stop_loss_pct=self.config.stop_loss_pct,
            volatility=volatility,
            **kwargs
        )
    
    def check_stop_loss(self, position) -> bool:
        """
        Check if position should be exited due to stop loss.
        
        Args:
            position: Position object
            
        Returns:
            True if should exit, False otherwise
        """
        if self.config.stop_loss_pct is None:
            return False
        return position.pnl_pct <= -self.config.stop_loss_pct
    
    def check_take_profit(self, position) -> bool:
        """
        Check if position should be exited due to take profit.
        
        Args:
            position: Position object
            
        Returns:
            True if should exit, False otherwise
        """
        if self.config.take_profit_pct is None:
            return False
        return position.pnl_pct >= self.config.take_profit_pct
    
    def check_concentration_limit(
        self, 
        ticker: str, 
        new_position_value: float,
        portfolio_value: float
    ) -> bool:
        """
        Check if new position would violate concentration limits.
        
        Args:
            ticker: Asset ticker
            new_position_value: Value of proposed position
            portfolio_value: Current portfolio value
            
        Returns:
            True if within limits, False if would violate
        """
        if portfolio_value == 0:
            return True
        
        concentration = new_position_value / portfolio_value
        return concentration <= self.config.max_position_size
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of portfolio returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR as a negative percentage
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Args:
            returns: Series of portfolio returns
            confidence: Confidence level
            
        Returns:
            Average return in worst (1-confidence)% of cases
        """
        if len(returns) == 0:
            return 0.0
        
        var_threshold = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return tail_returns.mean()
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.
        
        Args:
            win_rate: Probability of winning trade (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Kelly fraction (typically use half-Kelly in practice)
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Return half-Kelly for safety (common practice)
        return max(0, kelly * 0.5)
    
    # =========================================================================
    # KILL SWITCHES AND CIRCUIT BREAKERS
    # =========================================================================
    
    def initialize_capital_tracking(self, initial_capital: float, current_date: date = None):
        """Initialize capital tracking for kill switches."""
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.last_reset_date = current_date or date.today()
    
    def update_capital(self, current_capital: float, current_date: date = None):
        """
        Update capital tracking and check kill switches.
        
        Args:
            current_capital: Current total portfolio value
            current_date: Date for daily tracking
            
        Returns:
            Dict with breach information
        """
        if current_date is None:
            current_date = date.today()
        
        if self.initial_capital is None:
            self.initialize_capital_tracking(current_capital, current_date)
            return {'breaches': [], 'is_killed': False}
        
        # Update peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Reset daily tracking if new day
        if current_date != self.last_reset_date:
            self.daily_start_capital = current_capital
            self.last_reset_date = current_date
        
        breaches = []
        
        # Check kill switches
        current_dd = self.get_current_drawdown(current_capital)
        if current_dd >= self.config.max_drawdown_pct:
            self._trigger_kill_switch(
                f"Max drawdown {current_dd:.1%} reached (limit: {self.config.max_drawdown_pct:.1%})"
            )
            breaches.append({
                'type': 'KILL_SWITCH',
                'rule': 'max_drawdown',
                'value': current_dd,
                'limit': self.config.max_drawdown_pct,
                'timestamp': datetime.now()
            })
        
        daily_pnl = self.get_daily_pnl_pct(current_capital)
        if daily_pnl <= -self.config.max_daily_loss_pct:
            self._trigger_kill_switch(
                f"Daily loss {daily_pnl:.1%} reached (limit: {-self.config.max_daily_loss_pct:.1%})"
            )
            breaches.append({
                'type': 'KILL_SWITCH',
                'rule': 'max_daily_loss',
                'value': daily_pnl,
                'limit': -self.config.max_daily_loss_pct,
                'timestamp': datetime.now()
            })
        
        capital_pct = current_capital / self.initial_capital
        if capital_pct < self.config.min_capital_pct:
            self._trigger_kill_switch(
                f"Capital {capital_pct:.1%} of initial, below minimum {self.config.min_capital_pct:.1%}"
            )
            breaches.append({
                'type': 'KILL_SWITCH',
                'rule': 'min_capital',
                'value': capital_pct,
                'limit': self.config.min_capital_pct,
                'timestamp': datetime.now()
            })
        
        if breaches:
            self.breach_history.extend(breaches)
        
        return {
            'breaches': breaches,
            'is_killed': self.is_killed,
            'kill_reason': self.kill_reason,
            'stats': {
                'drawdown': current_dd,
                'daily_pnl_pct': daily_pnl,
                'capital_pct': capital_pct
            }
        }
    
    def check_trade_approval(
        self,
        asset: str,
        size: float,
        price: float,
        current_positions: Optional[Dict] = None,
        portfolio_value: float = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if trade passes all risk limits BEFORE execution.
        
        Args:
            asset: Asset ticker
            size: Position size
            price: Current price
            current_positions: Dict of current positions
            portfolio_value: Current portfolio value
            
        Returns:
            (approved, reasons) - True if approved, list of blocking reasons if not
        """
        if self.is_killed:
            return False, [f"KILL SWITCH ACTIVE: {self.kill_reason}"]
        
        if portfolio_value is None or portfolio_value <= 0:
            return False, ["Portfolio value not set or invalid"]
        
        reasons = []
        
        # Check position size limit
        notional = abs(size * price)
        position_pct = notional / portfolio_value
        
        if position_pct > self.config.max_position_size:
            reasons.append(
                f"Position size {position_pct:.1%} exceeds limit {self.config.max_position_size:.1%}"
            )
        
        # Check if adding to losing position
        if current_positions and asset in current_positions:
            pos = current_positions[asset]
            if 'pnl_pct' in pos and pos['pnl_pct'] < -0.05:  # -5% loss
                reasons.append(
                    f"Position {asset} already down {pos['pnl_pct']:.1%}"
                )
        
        return len(reasons) == 0, reasons
    
    def check_portfolio_heat(self, positions: Dict[str, Dict], portfolio_value: float) -> Dict:
        """
        Check portfolio heat (total at-risk capital).
        
        Args:
            positions: Dict of {asset: {pnl, pnl_pct, ...}}
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with heat metrics and breaches
        """
        if portfolio_value <= 0:
            return {'heat': 0.0, 'breaches': []}
        
        # Sum unrealized losses
        at_risk = sum(abs(pos.get('pnl', 0)) for pos in positions.values() 
                     if pos.get('pnl', 0) < 0)
        heat = at_risk / portfolio_value
        
        breaches = []
        if heat > self.config.max_portfolio_heat_pct:
            breaches.append({
                'type': 'PORTFOLIO_LIMIT',
                'rule': 'max_portfolio_heat',
                'value': heat,
                'limit': self.config.max_portfolio_heat_pct,
                'action': 'REDUCE_RISK',
                'timestamp': datetime.now()
            })
        
        return {'heat': heat, 'breaches': breaches}
    
    def get_current_drawdown(self, current_capital: float) -> float:
        """Get current drawdown from peak."""
        if self.peak_capital is None or self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - current_capital) / self.peak_capital
    
    def get_daily_pnl_pct(self, current_capital: float) -> float:
        """Get today's P&L percentage."""
        if self.daily_start_capital is None or self.daily_start_capital == 0:
            return 0.0
        return (current_capital - self.daily_start_capital) / self.daily_start_capital
    
    def _trigger_kill_switch(self, reason: str):
        """Activate kill switch - halts all trading."""
        if not self.is_killed:
            self.is_killed = True
            self.kill_reason = reason
            print(f"\n{'='*80}")
            print(f"🚨 KILL SWITCH ACTIVATED 🚨")
            print(f"{'='*80}")
            print(f"Reason: {reason}")
            print(f"Time: {datetime.now()}")
            print(f"{'='*80}")
    
    def reset_kill_switch(self, reason: str = "Manual override"):
        """Reset kill switch (requires manual confirmation)."""
        if self.is_killed:
            print(f"\n⚠️  Resetting kill switch...")
            print(f"   Previous reason: {self.kill_reason}")
            print(f"   Reset reason: {reason}")
            self.is_killed = False
            self.kill_reason = None
    
    def print_risk_status(self, current_capital: float = None, positions: Dict = None):
        """Print current risk status."""
        print("\n" + "="*80)
        print("📊 RISK STATUS")
        print("="*80)
        
        if self.is_killed:
            print(f"🚨 KILL SWITCH: ACTIVE")
            print(f"   Reason: {self.kill_reason}")
        else:
            print(f"✅ Kill Switch: Inactive")
        
        if current_capital and self.initial_capital:
            print(f"\n💰 Capital:")
            print(f"   Current:   ${current_capital:>12,.2f}")
            if self.peak_capital:
                print(f"   Peak:      ${self.peak_capital:>12,.2f}")
                print(f"   Drawdown:  {self.get_current_drawdown(current_capital):>12.1%}")
            if self.daily_start_capital:
                print(f"   Daily P&L: {self.get_daily_pnl_pct(current_capital):>12.1%}")
        
        print(f"\n⚠️  Risk Limits:")
        print(f"   Max Drawdown:     {self.config.max_drawdown_pct:.1%}")
        print(f"   Max Daily Loss:   {self.config.max_daily_loss_pct:.1%}")
        print(f"   Max Position:     {self.config.max_position_size:.1%}")
        print(f"   Max Heat:         {self.config.max_portfolio_heat_pct:.1%}")
        
        if positions:
            print(f"\n📈 Portfolio:")
            print(f"   Positions:      {len(positions):>8}")
        
        print("="*80)

