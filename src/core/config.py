"""
Configuration for MT5 Linux client
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MT5Config:
    """MT5 client configuration"""
    
    # Windows MT5 server settings
    SERVER_HOST: str = os.getenv('MT5_SERVER_HOST', '192.168.0.102')  # Windows MT5 server IP
    SERVER_PORT: int = int(os.getenv('MT5_SERVER_PORT', '5000'))
    API_KEY: Optional[str] = os.getenv('MT5_API_KEY')  # Optional API key for authentication
    
    # Trading parameters
    DEFAULT_VOLUME: float = float(os.getenv('MT5_DEFAULT_VOLUME', '0.01'))  # Default trade size in lots
    DEFAULT_STOP_LOSS_PIPS: int = int(os.getenv('MT5_DEFAULT_SL_PIPS', '50'))
    DEFAULT_TAKE_PROFIT_PIPS: int = int(os.getenv('MT5_DEFAULT_TP_PIPS', '100'))
    
    # Symbols to trade
    SYMBOLS: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    
    # Timeframes for analysis
    TIMEFRAMES: List[str] = field(default_factory=lambda: ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'])
    
    @property
    def server_url(self) -> str:
        """Get full server URL"""
        return f"http://{self.SERVER_HOST}:{self.SERVER_PORT}"

# Create default config instance
config = MT5Config()