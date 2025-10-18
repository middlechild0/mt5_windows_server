"""
Real-time market data streaming and processing.
"""
from typing import Dict, List, Optional, Callable, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import asyncio
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
import time

from src.core.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)

@dataclass
class MarketUpdate:
    """Single market data update."""
    pair: str
    timestamp: datetime
    bid: float
    ask: float
    volume: float = 0.0

class MarketDataStream:
    def __init__(self, pairs: List[str], mt5_connector: MT5Connector, buffer_size: int = 10000):
        """
        Initialize market data stream.
        
        Args:
            pairs: List of currency pairs to track
            mt5_connector: Initialized MT5 connector
            buffer_size: Maximum number of updates to keep in memory
        """
        self.pairs = set(pairs)
        self.mt5 = mt5_connector
        self.buffer_size = buffer_size
        self.data_buffers: Dict[str, pd.DataFrame] = {}
        self.ohlcv_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.callbacks: List[Callable] = []
        self.running = False
        self.lock = Lock()
        self.data_thread = None
        
        # Initialize buffers
        for pair in pairs:
            self.data_buffers[pair] = pd.DataFrame(
                columns=['timestamp', 'bid', 'ask', 'volume']
            ).set_index('timestamp')
            
            # Initialize OHLCV data for different timeframes
            self.ohlcv_data[pair] = {
                '1m': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
                '5m': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
                '15m': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
                '1h': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
                '4h': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']),
                '1d': pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            }
    
    def start(self) -> None:
        """Start market data stream."""
        if not self.mt5.connected:
            raise RuntimeError("MT5 connection not initialized")
            
        self.running = True
        self.data_thread = Thread(target=self._stream_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        logger.info("Market data stream started")
    
    def stop(self) -> None:
        """Stop market data stream."""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=5.0)
        logger.info("Market data stream stopped")
        
    def _stream_data(self) -> None:
        """Background thread for streaming market data."""
        while self.running:
            try:
                for pair in self.pairs:
                    # Get latest tick data
                    df = self.mt5.get_real_time_data(pair, '1m', 1)
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        update = MarketUpdate(
                            pair=pair,
                            timestamp=df.index[-1].to_pydatetime(),
                            bid=latest['Close'] - 0.00010,  # Simulate bid/ask spread
                            ask=latest['Close'] + 0.00010,
                            volume=latest['Volume']
                        )
                        self.process_update(update)
                        
                time.sleep(1.0)  # Stream every second
                
            except Exception as e:
                logger.error(f"Error in market data stream: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for market updates.
        
        Args:
            callback: Function to call on market updates
        """
        self.callbacks.append(callback)
    
    def process_update(self, update: MarketUpdate) -> None:
        """
        Process a new market data update.
        
        Args:
            update: Market update data
        """
        if not self.running or update.pair not in self.pairs:
            return
            
        with self.lock:
            # Add to tick buffer
            self.data_buffers[update.pair].loc[update.timestamp] = [
                update.bid,
                update.ask,
                update.volume
            ]
            
            # Maintain buffer size
            if len(self.data_buffers[update.pair]) > self.buffer_size:
                self.data_buffers[update.pair] = self.data_buffers[update.pair].iloc[-self.buffer_size:]
            
            # Update OHLCV data
            self._update_ohlcv(update)
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in market data callback: {e}")
    
    def get_latest_data(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get latest OHLCV data for a pair and timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            
        Returns:
            DataFrame with OHLCV data
        """
        with self.lock:
            if pair not in self.ohlcv_data or timeframe not in self.ohlcv_data[pair]:
                return None
            return self.ohlcv_data[pair][timeframe].copy()
    
    def _update_ohlcv(self, update: MarketUpdate) -> None:
        """
        Update OHLCV data with new tick.
        
        Args:
            update: Market update data
        """
        price = (update.bid + update.ask) / 2
        
        # Update all timeframes
        for timeframe, data in self.ohlcv_data[update.pair].items():
            period_start = self._get_period_start(update.timestamp, timeframe)
            
            # If new period, create new candle
            if len(data) == 0 or period_start not in data.index:
                # Delete older candles to maintain a clean OHLCV history
                data.drop(data.index[data.index < period_start], inplace=True)
                
                mid = (update.bid + update.ask) / 2
                data.loc[period_start] = {
                    'open': mid,
                    'high': mid,
                    'low': mid,
                    'close': mid,
                    'volume': update.volume
                }
            else:
                # Update existing candle using current prices
                mid = (update.bid + update.ask) / 2
                data.loc[period_start, 'high'] = max(data.loc[period_start, 'high'], mid)
                data.loc[period_start, 'low'] = min(data.loc[period_start, 'low'], mid)
                data.loc[period_start, 'close'] = mid
                data.loc[period_start, 'volume'] += update.volume
    
    @staticmethod
    def _get_period_start(timestamp: datetime, timeframe: str) -> datetime:
        """Calculate period start time for a given timestamp and timeframe."""
        if timeframe == '1m':
            return timestamp.replace(second=0, microsecond=0)
        elif timeframe == '5m':
            minutes = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
        elif timeframe == '15m':
            minutes = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
        elif timeframe == '1h':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == '4h':
            hours = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hours, minute=0, second=0, microsecond=0)
        elif timeframe == '1d':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")