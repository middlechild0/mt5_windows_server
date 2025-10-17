"""
Base strategy class that defines the interface for all trading strategies.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

@dataclass
class Signal:
    """Trading signal with risk management parameters."""
    pair: str
    timeframe: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    confidence: float = 0.0
    volume: Optional[float] = None
    signal_type: Optional[str] = None
    metadata: Optional[Dict] = None

class StrategyBase(ABC):
    def __init__(self, name: str, timeframes: List[str], pairs: List[str]):
        self.name = name
        self.timeframes = timeframes
        self.pairs = pairs
        self.signal_history: List[Signal] = []
        self.active_signals: Dict[str, Signal] = {}  # pair -> signal

    @abstractmethod
    def analyze(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> List[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: Dictionary of dataframes with OHLCV data for each timeframe
            current_time: Current timestamp for analysis
            
        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def validate_signal(self, signal: Signal, data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Validate a trading signal against current market conditions.
        
        Args:
            signal: Signal to validate
            data: Market data for validation
            
        Returns:
            (is_valid, reason)
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, balance: float, risk_per_trade: float) -> float:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            balance: Current account balance
            risk_per_trade: Maximum risk percentage per trade
            
        Returns:
            Position size in base currency
        """
        pass

    def update_signal(self, signal: Signal, current_data: pd.DataFrame) -> Optional[Signal]:
        """
        Update signal based on new market data.
        
        Args:
            signal: Existing signal
            current_data: Latest market data
            
        Returns:
            Updated signal or None if signal should be closed
        """
        # Default implementation - override for custom logic
        if signal.direction == 'buy':
            if current_data['low'].iloc[-1] <= signal.stop_loss:
                return None  # Stop loss hit
            if current_data['high'].iloc[-1] >= signal.take_profit:
                return None  # Take profit hit
        else:  # sell
            if current_data['high'].iloc[-1] >= signal.stop_loss:
                return None  # Stop loss hit
            if current_data['low'].iloc[-1] <= signal.take_profit:
                return None  # Take profit hit
        return signal

    def log_signal(self, signal: Signal) -> None:
        """Log generated signal for tracking."""
        self.signal_history.append(signal)
        self.active_signals[signal.pair] = signal