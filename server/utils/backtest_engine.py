"""
Backtesting engine for strategy testing and optimization.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

from src.core.strategy_base import Signal, StrategyBase
from src.core.risk_manager import RiskManager, RiskProfile
from src.core.trade_executor import TradeExecutor
from src.core.performance_monitor import PerformanceMonitor
from src.core.trade_tracker import TradeTracker

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    initial_balance: float
    pairs: List[str]
    timeframes: List[str]
    start_date: datetime
    end_date: datetime
    risk_profile: RiskProfile
    commission_rate: float = 0.001  # 0.1% per trade
    slippage: float = 0.0001  # 1 pip slippage

class BacktestEngine:
    def __init__(self, config: BacktestConfig, strategy: StrategyBase):
        self.config = config
        self.strategy = strategy
        self.current_balance = config.initial_balance
        
        # Initialize components
        self.risk_manager = RiskManager(config.risk_profile)
        self.trade_tracker = TradeTracker()
        self.trade_executor = TradeExecutor(self.risk_manager, self.trade_tracker)
        self.performance_monitor = PerformanceMonitor()
        
        # State tracking
        self.current_time: Optional[datetime] = None
        self.market_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
    def load_data(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """Load and validate market data for backtesting."""
        for pair in self.config.pairs:
            if pair not in data:
                raise ValueError(f"Missing data for pair {pair}")
            
            self.market_data[pair] = {}
            for tf in self.config.timeframes:
                if tf not in data[pair]:
                    raise ValueError(f"Missing {tf} timeframe data for {pair}")
                
                # Validate data structure
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data[pair][tf].columns for col in required_cols):
                    raise ValueError(f"Missing required columns in {pair} {tf} data")
                
                # Ensure data is sorted and indexed properly
                df = data[pair][tf].sort_index()
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"Index must be DatetimeIndex for {pair} {tf}")
                
                self.market_data[pair][tf] = df
    
    def run(self) -> None:
        """Run the backtest."""
        if not self.market_data:
            raise ValueError("No data loaded for backtest")
        
        # Get common timeframe for iteration
        main_pair = self.config.pairs[0]
        main_tf = self.config.timeframes[0]
        timeline = self.market_data[main_pair][main_tf].index
        
        # Filter timeline to backtest period
        mask = (timeline >= self.config.start_date) & (timeline <= self.config.end_date)
        timeline = timeline[mask]
        
        for timestamp in timeline:
            self.current_time = timestamp
            self._process_timestamp(timestamp)
            
        logger.info("Backtest completed")
    
    def _process_timestamp(self, timestamp: datetime) -> None:
        """Process a single timestamp in the backtest."""
        # Get current market data snapshot
        current_data = self._get_data_snapshot(timestamp)
        
        # Update existing positions
        current_prices = {pair: data[self.config.timeframes[0]]['close'].iloc[-1] 
                        for pair, data in current_data.items()}
        closed_positions = self.trade_executor.update_positions(current_prices, timestamp)
        
        # Generate new signals
        signals = self.strategy.analyze(current_data, timestamp)
        
        # Process each signal
        for signal in signals:
            # Skip if we already have a position for this pair
            if signal.pair in self.trade_executor.positions:
                continue
            
            # Validate signal with risk manager
            correlations = self._calculate_correlations(signal.pair)
            is_valid, reason = self.risk_manager.validate_trade(
                signal.pair, signal.direction, signal.entry_price,
                signal.stop_loss, signal.take_profit, timestamp,
                self.current_balance, correlations
            )
            
            if not is_valid:
                logger.debug(f"Signal rejected: {reason}")
                continue
            
            # Calculate position size
            volume = self.strategy.calculate_position_size(
                signal, self.current_balance,
                self.config.risk_profile.max_risk_per_trade
            )
            
            # Place order
            success, order_id = self.trade_executor.place_order(signal, volume)
            if success:
                # Simulate immediate fill with slippage
                fill_price = self._apply_slippage(signal.entry_price, signal.direction)
                self.trade_executor.update_order(order_id, 'filled', fill_price)
                
                # Track trade entry
                self.trade_tracker.record_entry(
                    order_id, signal.pair, self.current_time,
                    fill_price, volume, signal.direction,
                    signal.stop_loss, signal.take_profit
                )
                
                # Apply commission
                commission = fill_price * volume * self.config.commission_rate
                self.current_balance -= commission
    
    def _get_data_snapshot(self, timestamp: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get market data up to the current timestamp."""
        snapshot = {}
        for pair in self.config.pairs:
            snapshot[pair] = {}
            for tf in self.config.timeframes:
                df = self.market_data[pair][tf]
                snapshot[pair][tf] = df[df.index <= timestamp]
        return snapshot
    
    def _calculate_correlations(self, pair: str) -> Dict[str, float]:
        """Calculate correlations between the given pair and active positions."""
        correlations = {}
        main_tf = self.config.timeframes[0]
        
        for pos_pair in self.trade_executor.positions:
            if pos_pair == pair:
                continue
                
            # Calculate correlation over last 100 periods
            pair_data = self.market_data[pair][main_tf]['close'].tail(100)
            pos_data = self.market_data[pos_pair][main_tf]['close'].tail(100)
            
            correlation = pair_data.corr(pos_data)
            correlations[pos_pair] = correlation
        
        return correlations
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage to the execution price."""
        if direction == 'buy':
            return price * (1 + self.config.slippage)
        return price * (1 - self.config.slippage)
    
    def get_results(self) -> Dict:
        """Get backtest results and statistics."""
        stats = self.performance_monitor.calculate_stats(
            self.config.start_date,
            self.config.end_date
        )
        
        trades = self.trade_tracker.get_trades()
        
        # Create equity curve even if no trades
        equity_curve_data = []
        if trades:
            for t in trades:
                equity_curve_data.append({
                    'timestamp': t['exit_time'],
                    'equity': self.config.initial_balance + sum(tr['pnl'] for tr in trades if tr['exit_time'] <= t['exit_time']),
                    'trade_pnl': t['pnl']
                })
        else:
            # If no trades, create single point with initial balance
            equity_curve_data.append({
                'timestamp': self.config.end_date,
                'equity': self.config.initial_balance,
                'trade_pnl': 0
            })
            
        equity_curve = pd.DataFrame(equity_curve_data)
        
        return {
            'statistics': stats,
            'equity_curve': equity_curve,
            'daily_stats': self.performance_monitor.get_daily_stats(),
            'final_balance': self.current_balance,
            'total_return': (self.current_balance - self.config.initial_balance) / 
                           self.config.initial_balance * 100,
            'trades': trades
        }