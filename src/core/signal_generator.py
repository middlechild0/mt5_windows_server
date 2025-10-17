"""
Trading signal generator and manager.
"""
from datetime import datetime
from typing import Dict, List
import pandas as pd
from src.core.strategy_base import Signal, StrategyBase

class SignalGenerator:
    def __init__(self, strategies: List[StrategyBase]):
        self.strategies = strategies
        self.active_signals: Dict[str, Signal] = {}  # pair -> active signal
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]], current_time: datetime) -> List[Signal]:
        """
        Generate trading signals from all strategies.
        
        Args:
            data: {pair: {timeframe: DataFrame}} nested dict of OHLCV data
            current_time: Current market time
            
        Returns:
            List of validated signals
        """
        all_signals = []
        
        for strategy in self.strategies:
            # Prepare data for strategy's timeframes
            strategy_data = {}
            for tf in strategy.timeframes:
                for pair in strategy.pairs:
                    if pair in data and tf in data[pair]:
                        if pair not in strategy_data:
                            strategy_data[pair] = {}
                        strategy_data[pair][tf] = data[pair][tf]
            
            # Generate signals
            signals = strategy.analyze(strategy_data, current_time)
            
            # Validate each signal
            for signal in signals:
                is_valid, reason = strategy.validate_signal(signal, strategy_data)
                if is_valid:
                    all_signals.append(signal)
        
        return all_signals
    
    def update_active_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> List[Signal]:
        """
        Update status of active signals and remove closed ones.
        
        Args:
            data: Latest market data
            current_time: Current market time
            
        Returns:
            List of signals that should be closed
        """
        signals_to_close = []
        for pair, signal in list(self.active_signals.items()):
            if pair in data:
                updated_signal = signal.strategy.update_signal(signal, data[pair])
                if updated_signal is None:
                    signals_to_close.append(signal)
                    del self.active_signals[pair]
                else:
                    self.active_signals[pair] = updated_signal
        
        return signals_to_close