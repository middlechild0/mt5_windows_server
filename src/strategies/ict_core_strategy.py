"""
Basic ICT concepts implementation strategy.
"""
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from src.core.strategy_base import StrategyBase, Signal

class ICTCoreStrategy(StrategyBase):
    def __init__(self, name: str, timeframes: List[str], pairs: List[str]):
        super().__init__(name, timeframes, pairs)
        self.name = "ICT Core Strategy"
        
        # Strategy parameters
        self.atr_period = 14
        self.volume_ma_period = 20
        self.min_volume_multiplier = 1.5
        self.risk_reward_ratio = 2.0
        
    def analyze(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> List[Signal]:
        signals = []
        
        for pair, pair_data in data.items():
            # Get main timeframe data
            main_tf = self.timeframes[0]
            if main_tf not in pair_data or len(pair_data[main_tf]) < 50:
                continue
                
            df = pair_data[main_tf]
            current_candle = df.iloc[-1]
            
            # Calculate indicators
            atr = self._calculate_atr(df)
            volume_ma = df['volume'].rolling(self.volume_ma_period).mean()
            
            # Identify liquidity levels
            highs = self._find_liquidity_levels(df, 'high')
            lows = self._find_liquidity_levels(df, 'low')
            
            # Check for volume anomalies
            if current_candle['volume'] > volume_ma.iloc[-1] * self.min_volume_multiplier:
                # Check for price action at liquidity levels
                if self._is_liquidity_sweep(df, highs, 'high'):
                    signal = self._generate_sell_signal(pair, df, current_time, atr.iloc[-1])
                    if signal:
                        signals.append(signal)
                        
                elif self._is_liquidity_sweep(df, lows, 'low'):
                    signal = self._generate_buy_signal(pair, df, current_time, atr.iloc[-1])
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def validate_signal(self, signal: Signal, data: Dict[str, pd.DataFrame]) -> Tuple[bool, str]:
        if signal.pair not in data:
            return False, "No data available for pair"
            
        df = data[signal.pair][self.timeframes[0]]
        current_price = df['close'].iloc[-1]
        
        # Validate signal is still within acceptable range
        if signal.direction == 'buy':
            if current_price > signal.entry_price * 1.001:  # 0.1% threshold
                return False, "Price moved too far from entry"
        else:
            if current_price < signal.entry_price * 0.999:
                return False, "Price moved too far from entry"
        
        return True, "Signal validated"
    
    def calculate_position_size(self, signal: Signal, balance: float, risk_per_trade: float) -> float:
        """Calculate position size based on risk parameters."""
        risk_amount = balance * (risk_per_trade / 100)
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        
        if pip_risk == 0:
            return 0
            
        return risk_amount / pip_risk
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def _find_liquidity_levels(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Find significant liquidity levels using swing highs/lows."""
        period = 10
        if col == 'high':
            is_extreme = (df[col] > df[col].shift(period)) & (df[col] > df[col].shift(-period))
        else:
            is_extreme = (df[col] < df[col].shift(period)) & (df[col] < df[col].shift(-period))
            
        return df[col][is_extreme]
    
    def _is_liquidity_sweep(self, df: pd.DataFrame, levels: pd.Series, level_type: str) -> bool:
        """Check if recent price action indicates a liquidity sweep."""
        if len(levels) < 2:
            return False
            
        current_price = df['close'].iloc[-1]
        recent_levels = levels.tail(3)
        
        if level_type == 'high':
            # Check if price swept above recent high and reversed
            if current_price < recent_levels.iloc[-1] and df['high'].iloc[-1] > recent_levels.iloc[-1]:
                return True
        else:
            # Check if price swept below recent low and reversed
            if current_price > recent_levels.iloc[-1] and df['low'].iloc[-1] < recent_levels.iloc[-1]:
                return True
                
        return False
    
    def _generate_buy_signal(self, pair: str, df: pd.DataFrame, 
                           timestamp: datetime, atr: float) -> Optional[Signal]:
        """Generate buy signal with stop and targets based on ICT concepts."""
        entry_price = df['close'].iloc[-1]
        stop_loss = entry_price - atr * 1.5  # 1.5 ATR for stop loss
        take_profit = entry_price + atr * 1.5 * self.risk_reward_ratio
        
        return Signal(
            timestamp=timestamp,
            pair=pair,
            direction='buy',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type='ict_liquidity_sweep_long',
            confidence=0.8,
            timeframe=self.timeframes[0],
            metadata={'atr': atr}
        )
    
    def _generate_sell_signal(self, pair: str, df: pd.DataFrame,
                            timestamp: datetime, atr: float) -> Optional[Signal]:
        """Generate sell signal with stop and targets based on ICT concepts."""
        entry_price = df['close'].iloc[-1]
        stop_loss = entry_price + atr * 1.5  # 1.5 ATR for stop loss
        take_profit = entry_price - atr * 1.5 * self.risk_reward_ratio
        
        return Signal(
            timestamp=timestamp,
            pair=pair,
            direction='sell',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type='ict_liquidity_sweep_short',
            confidence=0.8,
            timeframe=self.timeframes[0],
            metadata={'atr': atr}
        )