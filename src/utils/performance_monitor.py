"""
Performance monitoring and analytics system.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    total_pnl: float
    risk_reward_ratio: float

class PerformanceMonitor:
    def __init__(self):
        self.trades_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.equity_curve: List[Dict] = []
        
    def add_trade(self, trade_data: Dict) -> None:
        """Add a completed trade to history."""
        self.trades_history.append(trade_data)
        
        # Update daily PnL
        date = trade_data['exit_time'].date().isoformat()
        if date not in self.daily_pnl:
            self.daily_pnl[date] = 0
        self.daily_pnl[date] += trade_data['pnl']
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': trade_data['exit_time'],
            'equity': sum(t['pnl'] for t in self.trades_history),
            'trade_pnl': trade_data['pnl']
        })
    
    def calculate_stats(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> TradeStats:
        """Calculate performance statistics for the given period."""
        # Filter trades by time period
        trades = self.trades_history
        if start_time:
            trades = [t for t in trades if t['exit_time'] >= start_time]
        if end_time:
            trades = [t for t in trades if t['exit_time'] <= end_time]
        
        if not trades:
            return None
        
        # Basic statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_trades = len(trades)
        total_wins = len(winning_trades)
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        total_pnl = sum(t['pnl'] for t in trades)
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate Sharpe Ratio (assuming daily returns)
        daily_returns = pd.Series(self.daily_pnl).values
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Calculate max drawdown
        equity_curve = pd.DataFrame(self.equity_curve)
        rolling_max = equity_curve['equity'].cummax()
        drawdowns = (rolling_max - equity_curve['equity']) / rolling_max * 100
        max_drawdown = drawdowns.max()
        
        # Calculate max drawdown duration
        if len(equity_curve) > 1:
            drawdown_periods = equity_curve['timestamp'].diff().fillna(pd.Timedelta(0))
            max_drawdown_duration = drawdown_periods.max()
        else:
            max_drawdown_duration = timedelta(0)
        
        # Calculate average risk/reward ratio
        risk_reward_ratios = [(t['take_profit'] - t['entry_price']) / (t['entry_price'] - t['stop_loss']) 
                            if t['direction'] == 'buy' else
                            (t['entry_price'] - t['take_profit']) / (t['stop_loss'] - t['entry_price'])
                            for t in trades]
        avg_risk_reward = np.mean(risk_reward_ratios)
        
        return TradeStats(
            total_trades=total_trades,
            winning_trades=total_wins,
            losing_trades=total_trades - total_wins,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_pnl=total_pnl,
            risk_reward_ratio=avg_risk_reward
        )
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame."""
        return pd.DataFrame(self.equity_curve)
    
    def get_daily_stats(self) -> pd.DataFrame:
        """Calculate daily performance metrics."""
        df = pd.DataFrame.from_dict(self.daily_pnl, orient='index', columns=['pnl'])
        df.index = pd.to_datetime(df.index)
        
        # Calculate cumulative metrics
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['rolling_sharpe'] = (df['pnl'].rolling(window=20).mean() / 
                              df['pnl'].rolling(window=20).std() * 
                              np.sqrt(252))
        
        return df