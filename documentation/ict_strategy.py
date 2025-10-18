#!/usr/bin/env python3
"""
CLEAN ICT TRADING STRATEGY
=========================

Professional ICT (Inner Circle Trader) methodology implementation
- Order Block detection
- Liquidity Sweep analysis
- Fair Value Gap identification
- Session-based trading

October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ICTSetup(Enum):
    """ICT trading setups"""
    ORDER_BLOCK_BULL = "order_block_bullish"
    ORDER_BLOCK_BEAR = "order_block_bearish"
    LIQUIDITY_SWEEP_HIGH = "liquidity_sweep_high"
    LIQUIDITY_SWEEP_LOW = "liquidity_sweep_low"
    FAIR_VALUE_GAP = "fair_value_gap"
    IMBALANCE = "imbalance"

@dataclass
class ICTSignal:
    """Professional ICT signal structure"""
    setup: ICTSetup
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    reason: str
    session: str

class ICTTradingStrategy:
    """Professional ICT trading methodology"""
    
    def __init__(self):
        """Initialize ICT strategy"""
        
        # ICT parameters
        self.order_block_lookback = 20
        self.liquidity_threshold = 0.618  # Fibonacci level
        self.imbalance_min_size = 5  # Minimum imbalance in pips
        
        # Risk management
        self.default_sl_pips = 15
        self.default_tp_pips = 30
        self.risk_reward_ratio = 2.0
        
        # Session definitions (UTC)
        self.sessions = {
            'asian': {'start': 0, 'end': 9},      # 00:00 - 09:00 UTC
            'london': {'start': 7, 'end': 16},    # 07:00 - 16:00 UTC  
            'newyork': {'start': 13, 'end': 22},  # 13:00 - 22:00 UTC
            'overlap': {'start': 13, 'end': 16}   # London/NY overlap
        }
        
        # Currency pair specifications
        self.pair_specs = {
            'EURUSD': {'pip_value': 0.0001, 'spread_typical': 0.8},
            'GBPUSD': {'pip_value': 0.0001, 'spread_typical': 1.2},
            'USDJPY': {'pip_value': 0.01, 'spread_typical': 1.0},
            'USDCAD': {'pip_value': 0.0001, 'spread_typical': 1.5},
            'AUDUSD': {'pip_value': 0.0001, 'spread_typical': 1.0}
        }
    
    def analyze_price_action(self, symbol: str, bars: pd.DataFrame) -> List[ICTSignal]:
        """Comprehensive ICT price action analysis"""
        
        if bars.empty or len(bars) < self.order_block_lookback:
            return []
        
        signals = []
        current_session = self.get_current_session()
        
        # 1. Order Block Analysis
        ob_signals = self._detect_order_blocks(symbol, bars, current_session)
        signals.extend(ob_signals)
        
        # 2. Liquidity Sweep Detection  
        ls_signals = self._detect_liquidity_sweeps(symbol, bars, current_session)
        signals.extend(ls_signals)
        
        # 3. Fair Value Gap Analysis
        fvg_signals = self._detect_fair_value_gaps(symbol, bars, current_session)
        signals.extend(fvg_signals)
        
        # 4. Imbalance Detection
        imb_signals = self._detect_imbalances(symbol, bars, current_session)
        signals.extend(imb_signals)
        
        # Filter and rank signals
        filtered_signals = self._filter_signals(signals)
        
        return sorted(filtered_signals, key=lambda x: x.confidence, reverse=True)
    
    def _detect_order_blocks(self, symbol: str, bars: pd.DataFrame, session: str) -> List[ICTSignal]:
        """Detect ICT Order Blocks"""
        
        signals = []
        
        if len(bars) < 10:
            return signals
        
        # Calculate swing highs and lows
        highs = bars['high'].rolling(window=5, center=True).max()
        lows = bars['low'].rolling(window=5, center=True).min()
        
        # Identify potential order blocks
        for i in range(10, len(bars) - 5):
            
            current_bar = bars.iloc[i]
            
            # Bullish Order Block (after low sweep)
            if (bars['low'].iloc[i] == lows.iloc[i] and  # Swing low
                bars['close'].iloc[i] > bars['open'].iloc[i] and  # Bullish candle
                bars['close'].iloc[i+1:i+4].min() > bars['low'].iloc[i]):  # Price holds above
                
                confidence = self._calculate_ob_confidence(bars, i, 'bullish')
                
                if confidence > 0.6:
                    entry_price = bars['close'].iloc[i]
                    sl_pips = self.default_sl_pips
                    tp_pips = self.default_tp_pips
                    
                    signal = ICTSignal(
                        setup=ICTSetup.ORDER_BLOCK_BULL,
                        symbol=symbol,
                        direction='BUY',
                        entry_price=entry_price,
                        stop_loss=entry_price - (sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price + (tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i],
                        reason=f"Bullish OB at {entry_price:.5f}",
                        session=session
                    )
                    signals.append(signal)
            
            # Bearish Order Block (after high sweep)
            if (bars['high'].iloc[i] == highs.iloc[i] and  # Swing high
                bars['close'].iloc[i] < bars['open'].iloc[i] and  # Bearish candle
                bars['close'].iloc[i+1:i+4].max() < bars['high'].iloc[i]):  # Price holds below
                
                confidence = self._calculate_ob_confidence(bars, i, 'bearish')
                
                if confidence > 0.6:
                    entry_price = bars['close'].iloc[i]
                    sl_pips = self.default_sl_pips
                    tp_pips = self.default_tp_pips
                    
                    signal = ICTSignal(
                        setup=ICTSetup.ORDER_BLOCK_BEAR,
                        symbol=symbol,
                        direction='SELL',
                        entry_price=entry_price,
                        stop_loss=entry_price + (sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price - (tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i],
                        reason=f"Bearish OB at {entry_price:.5f}",
                        session=session
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_liquidity_sweeps(self, symbol: str, bars: pd.DataFrame, session: str) -> List[ICTSignal]:
        """Detect ICT Liquidity Sweeps"""
        
        signals = []
        
        if len(bars) < 20:
            return signals
        
        # Look for liquidity grabs
        for i in range(15, len(bars) - 1):
            
            # High liquidity sweep (false breakout above resistance)
            recent_highs = bars['high'].iloc[i-15:i]
            resistance_level = recent_highs.max()
            
            if (bars['high'].iloc[i] > resistance_level and  # Break above
                bars['close'].iloc[i] < resistance_level):    # Close back below
                
                # Check for reversal
                if bars['close'].iloc[i+1] < bars['close'].iloc[i]:
                    
                    confidence = 0.75  # High confidence for liquidity sweeps
                    entry_price = bars['close'].iloc[i+1]
                    
                    signal = ICTSignal(
                        setup=ICTSetup.LIQUIDITY_SWEEP_HIGH,
                        symbol=symbol,
                        direction='SELL',
                        entry_price=entry_price,
                        stop_loss=entry_price + (self.default_sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price - (self.default_tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i+1],
                        reason=f"High liquidity sweep at {resistance_level:.5f}",
                        session=session
                    )
                    signals.append(signal)
            
            # Low liquidity sweep (false breakout below support)
            recent_lows = bars['low'].iloc[i-15:i]
            support_level = recent_lows.min()
            
            if (bars['low'].iloc[i] < support_level and    # Break below
                bars['close'].iloc[i] > support_level):    # Close back above
                
                # Check for reversal
                if bars['close'].iloc[i+1] > bars['close'].iloc[i]:
                    
                    confidence = 0.75
                    entry_price = bars['close'].iloc[i+1]
                    
                    signal = ICTSignal(
                        setup=ICTSetup.LIQUIDITY_SWEEP_LOW,
                        symbol=symbol,
                        direction='BUY',
                        entry_price=entry_price,
                        stop_loss=entry_price - (self.default_sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price + (self.default_tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i+1],
                        reason=f"Low liquidity sweep at {support_level:.5f}",
                        session=session
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_fair_value_gaps(self, symbol: str, bars: pd.DataFrame, session: str) -> List[ICTSignal]:
        """Detect ICT Fair Value Gaps"""
        
        signals = []
        
        for i in range(2, len(bars)):
            
            # Bullish FVG (gap up)
            if (bars['low'].iloc[i] > bars['high'].iloc[i-2] and
                bars['high'].iloc[i-1] < bars['low'].iloc[i]):
                
                gap_size = bars['low'].iloc[i] - bars['high'].iloc[i-2]
                gap_pips = gap_size / self.get_pip_value(symbol)
                
                if gap_pips >= 3:  # Minimum 3 pip gap
                    
                    confidence = min(0.8, gap_pips / 10)  # Higher confidence for larger gaps
                    entry_price = (bars['low'].iloc[i] + bars['high'].iloc[i-2]) / 2
                    
                    signal = ICTSignal(
                        setup=ICTSetup.FAIR_VALUE_GAP,
                        symbol=symbol,
                        direction='BUY',
                        entry_price=entry_price,
                        stop_loss=entry_price - (self.default_sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price + (self.default_tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i],
                        reason=f"Bullish FVG ({gap_pips:.1f} pips)",
                        session=session
                    )
                    signals.append(signal)
            
            # Bearish FVG (gap down)
            if (bars['high'].iloc[i] < bars['low'].iloc[i-2] and
                bars['low'].iloc[i-1] > bars['high'].iloc[i]):
                
                gap_size = bars['low'].iloc[i-2] - bars['high'].iloc[i]
                gap_pips = gap_size / self.get_pip_value(symbol)
                
                if gap_pips >= 3:
                    
                    confidence = min(0.8, gap_pips / 10)
                    entry_price = (bars['high'].iloc[i] + bars['low'].iloc[i-2]) / 2
                    
                    signal = ICTSignal(
                        setup=ICTSetup.FAIR_VALUE_GAP,
                        symbol=symbol,
                        direction='SELL',
                        entry_price=entry_price,
                        stop_loss=entry_price + (self.default_sl_pips * self.get_pip_value(symbol)),
                        take_profit=entry_price - (self.default_tp_pips * self.get_pip_value(symbol)),
                        confidence=confidence,
                        timestamp=bars.index[i],
                        reason=f"Bearish FVG ({gap_pips:.1f} pips)",
                        session=session
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_imbalances(self, symbol: str, bars: pd.DataFrame, session: str) -> List[ICTSignal]:
        """Detect market imbalances"""
        
        signals = []
        
        for i in range(1, len(bars)):
            
            # Calculate candle body and wicks
            body_size = abs(bars['close'].iloc[i] - bars['open'].iloc[i])
            upper_wick = bars['high'].iloc[i] - max(bars['open'].iloc[i], bars['close'].iloc[i])
            lower_wick = min(bars['open'].iloc[i], bars['close'].iloc[i]) - bars['low'].iloc[i]
            
            body_pips = body_size / self.get_pip_value(symbol)
            
            # Strong bullish imbalance
            if (bars['close'].iloc[i] > bars['open'].iloc[i] and
                body_pips >= self.imbalance_min_size and
                upper_wick < body_size * 0.2 and
                lower_wick < body_size * 0.2):
                
                confidence = min(0.7, body_pips / 20)
                entry_price = bars['close'].iloc[i]
                
                signal = ICTSignal(
                    setup=ICTSetup.IMBALANCE,
                    symbol=symbol,
                    direction='BUY',
                    entry_price=entry_price,
                    stop_loss=entry_price - (self.default_sl_pips * self.get_pip_value(symbol)),
                    take_profit=entry_price + (self.default_tp_pips * self.get_pip_value(symbol)),
                    confidence=confidence,
                    timestamp=bars.index[i],
                    reason=f"Bullish imbalance ({body_pips:.1f} pips)",
                    session=session
                )
                signals.append(signal)
            
            # Strong bearish imbalance
            if (bars['close'].iloc[i] < bars['open'].iloc[i] and
                body_pips >= self.imbalance_min_size and
                upper_wick < body_size * 0.2 and
                lower_wick < body_size * 0.2):
                
                confidence = min(0.7, body_pips / 20)
                entry_price = bars['close'].iloc[i]
                
                signal = ICTSignal(
                    setup=ICTSetup.IMBALANCE,
                    symbol=symbol,
                    direction='SELL',
                    entry_price=entry_price,
                    stop_loss=entry_price + (self.default_sl_pips * self.get_pip_value(symbol)),
                    take_profit=entry_price - (self.default_tp_pips * self.get_pip_value(symbol)),
                    confidence=confidence,
                    timestamp=bars.index[i],
                    reason=f"Bearish imbalance ({body_pips:.1f} pips)",
                    session=session
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_ob_confidence(self, bars: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate order block confidence"""
        
        confidence = 0.5  # Base confidence
        
        # Volume confirmation (if available)
        if 'volume' in bars.columns:
            avg_volume = bars['volume'].iloc[index-10:index].mean()
            current_volume = bars['volume'].iloc[index]
            
            if current_volume > avg_volume * 1.5:
                confidence += 0.2
        
        # Candle size confirmation
        body_size = abs(bars['close'].iloc[index] - bars['open'].iloc[index])
        candle_range = bars['high'].iloc[index] - bars['low'].iloc[index]
        
        if body_size > candle_range * 0.7:  # Strong candle
            confidence += 0.1
        
        # Session timing
        current_hour = bars.index[index].hour
        
        # Higher confidence during active sessions
        if (7 <= current_hour <= 16 or  # London session
            13 <= current_hour <= 22):   # NY session
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _filter_signals(self, signals: List[ICTSignal]) -> List[ICTSignal]:
        """Filter and validate signals"""
        
        filtered = []
        
        for signal in signals:
            # Minimum confidence threshold
            if signal.confidence < 0.6:
                continue
            
            # Risk-reward validation
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            
            if reward / risk < 1.5:  # Minimum 1.5:1 RR
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def get_current_session(self) -> str:
        """Get current trading session"""
        
        current_hour = datetime.utcnow().hour
        
        # Check for overlap first (highest priority)
        if self.sessions['overlap']['start'] <= current_hour <= self.sessions['overlap']['end']:
            return 'overlap'
        
        # London session
        elif self.sessions['london']['start'] <= current_hour <= self.sessions['london']['end']:
            return 'london'
        
        # New York session  
        elif self.sessions['newyork']['start'] <= current_hour <= self.sessions['newyork']['end']:
            return 'newyork'
        
        # Asian session
        else:
            return 'asian'
    
    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        
        # Try to get from specifications
        if symbol in self.pair_specs:
            return self.pair_specs[symbol]['pip_value']
        
        # Default logic
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def get_session_bias(self, session: str) -> str:
        """Get session trading bias"""
        
        session_bias = {
            'asian': 'range',      # Range-bound trading
            'london': 'trend',     # Trend following
            'newyork': 'trend',    # Trend following  
            'overlap': 'breakout'  # Breakout trading
        }
        
        return session_bias.get(session, 'neutral')

def test_ict_strategy():
    """Test ICT strategy"""
    
    print("🧪 Testing ICT Trading Strategy")
    print("=" * 40)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Generate realistic EURUSD data
    np.random.seed(42)
    prices = []
    base_price = 1.1000
    
    for i in range(100):
        change = np.random.normal(0, 0.0005)  # Small random changes
        base_price += change
        prices.append(base_price)
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.0003)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.0003)) for p in prices],
        'close': [p + np.random.normal(0, 0.0002) for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure OHLC relationship is correct
    for i in range(len(sample_data)):
        high = max(sample_data.iloc[i][['open', 'close']].max(), sample_data.iloc[i]['high'])
        low = min(sample_data.iloc[i][['open', 'close']].min(), sample_data.iloc[i]['low'])
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = high
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = low
    
    # Initialize strategy
    ict = ICTTradingStrategy()
    
    # Analyze price action
    signals = ict.analyze_price_action('EURUSD', sample_data)
    
    print(f"📊 Found {len(signals)} ICT signals")
    
    for i, signal in enumerate(signals[:5], 1):
        print(f"\n{i}. {signal.setup.value}")
        print(f"   Direction: {signal.direction}")
        print(f"   Entry: {signal.entry_price:.5f}")
        print(f"   SL: {signal.stop_loss:.5f}")
        print(f"   TP: {signal.take_profit:.5f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Reason: {signal.reason}")
        print(f"   Session: {signal.session}")
    
    # Session info
    current_session = ict.get_current_session()
    session_bias = ict.get_session_bias(current_session)
    
    print(f"\n🕐 Current Session: {current_session}")
    print(f"📈 Session Bias: {session_bias}")
    
    print("\n✅ ICT strategy test complete!")

if __name__ == "__main__":
    test_ict_strategy()