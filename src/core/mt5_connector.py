"""
MetaTrader 5 integration for real-time trading.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    """MT5 connection configuration."""
    path: str = ""  # Path to MT5 terminal
    login: int = 0  # Account login
    password: str = ""  # Account password
    server: str = ""  # Trade server
    timeout: int = 10000  # Timeout in milliseconds

class MT5Connector:
    """MetaTrader 5 connection and trading interface."""
    
    def __init__(self, config: MT5Config):
        """
        Initialize MT5 connector.
        
        Args:
            config: MT5 configuration
        """
        self.config = config
        self.connected = False
        
    def connect(self) -> bool:
        """
        Initialize connection to MetaTrader 5 terminal.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize(
                path=self.config.path,
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            self.connected = True
            logger.info("Successfully connected to MT5")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
            
    def disconnect(self) -> None:
        """Shutdown connection to MetaTrader 5 terminal."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_real_time_data(self, 
                          symbol: str,
                          timeframe: str,
                          num_bars: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get real-time OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '4h', '1d')
            num_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        # Convert timeframe string to MT5 timeframe constant
        tf_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
        }
        
        if timeframe not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
            
        try:
            # Get bars from MT5
            bars = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, num_bars)
            if bars is None:
                logger.error(f"Failed to get data: {mt5.last_error()}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to match our format
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return None
    
    def place_market_order(self,
                          symbol: str,
                          order_type: str,
                          volume: float,
                          sl: float = 0.0,
                          tp: float = 0.0) -> Tuple[bool, str]:
        """
        Place a market order through MT5.
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            volume: Trade volume in lots
            sl: Stop loss price
            tp: Take profit price
            
        Returns:
            (success, order_id)
        """
        if not self.connected:
            return False, "Not connected to MT5"
            
        try:
            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not found"
                
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return False, f"Failed to select symbol {symbol}"
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False, f"Failed to get price for {symbol}"
                
            # Calculate points for deviation based on digits
            point = symbol_info.point
            digits = symbol_info.digits
            deviation = int(10 * (10 ** (digits - 5)))  # Adjust deviation based on digits
            
            # Prepare the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),  # Ensure volume is float
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": tick.ask if order_type == 'buy' else tick.bid,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": 234000,  # Magic number to identify our trades
                "comment": "ICT AI Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,  # Exness supports IOC filling
            }
            
            # Send the order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return False, result.comment
                
            return True, str(result.order)
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False, str(e)
    
    def get_account_info(self) -> Dict:
        """Get account information from MT5."""
        if not self.connected:
            return {}
            
        try:
            account_info = mt5.account_info()._asdict()
            return {
                'balance': account_info['balance'],
                'equity': account_info['equity'],
                'margin': account_info['margin'],
                'free_margin': account_info['margin_free'],
                'leverage': account_info['leverage'],
                'currency': account_info['currency']
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}