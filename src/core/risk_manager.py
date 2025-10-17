"""
Risk management system for trading operations.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

@dataclass
class RiskProfile:
    max_risk_per_trade: float  # percentage of account
    max_open_trades: int
    max_daily_loss: float  # percentage of account
    max_drawdown: float  # percentage from peak
    risk_reward_ratio: float
    correlation_limit: float  # maximum allowed correlation between pairs

class RiskManager:
    def __init__(self, risk_profile: RiskProfile):
        self.risk_profile = risk_profile
        self.daily_pnl = {}  # date -> PnL
        self.open_trades = {}  # pair -> trade info
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              account_balance: float) -> float:
        """Calculate position size based on risk parameters."""
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
            
        risk_amount = account_balance * (self.risk_profile.max_risk_per_trade / 100)
        pip_risk = abs(entry_price - stop_loss)
        
        if pip_risk == 0:
            return 0.0
            
        return risk_amount / pip_risk
    
    def validate_trade(self, pair: str, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, current_time: datetime,
                      account_balance: float, correlations: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk management criteria.
        
        Returns:
            (is_valid, reason)
        """
        # Check number of open trades
        if len(self.open_trades) >= self.risk_profile.max_open_trades:
            return False, "Maximum number of open trades reached"
        
        # Check daily loss limit
        date_key = current_time.date().isoformat()
        if date_key in self.daily_pnl:
            daily_loss = self.daily_pnl[date_key]
            max_daily_loss = account_balance * (self.risk_profile.max_daily_loss / 100)
            if daily_loss < -max_daily_loss:
                return False, "Daily loss limit reached"
        
        # Check drawdown
        if self.current_drawdown > self.risk_profile.max_drawdown:
            return False, "Maximum drawdown reached"
        
        # Check risk-reward ratio
        rr_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        if rr_ratio < self.risk_profile.risk_reward_ratio:
            return False, f"Risk-reward ratio {rr_ratio:.2f} below minimum {self.risk_profile.risk_reward_ratio}"
        
        # Check correlations
        for open_pair, correlation in correlations.items():
            if abs(correlation) > self.risk_profile.correlation_limit:
                return False, f"High correlation with open position in {open_pair}"
        
        return True, "Trade validated"
    
    def update_metrics(self, current_balance: float, current_time: datetime,
                      trade_pnl: float = 0.0) -> None:
        """Update risk metrics with new balance information."""
        # Update peak balance and drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        
        # Update daily PnL
        date_key = current_time.date().isoformat()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0.0
        self.daily_pnl[date_key] += trade_pnl