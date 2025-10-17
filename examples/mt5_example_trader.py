"""
Example trader using MT5 client to execute trades
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.core.mt5_client import MT5Client
from src.core.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExampleTrader:
    """Example trading class demonstrating MT5 client usage"""
    
    def __init__(self):
        """Initialize trader with MT5 client"""
        self.client = MT5Client(config.SERVER_HOST, config.SERVER_PORT)
        self.symbols = config.SYMBOLS
        
    def check_account(self) -> Optional[Dict]:
        """Check account status and balance"""
        account_info = self.client.get_account_info()
        if account_info:
            logger.info(f"Account Balance: {account_info.get('balance', 0)}")
            logger.info(f"Account Equity: {account_info.get('equity', 0)}")
            return account_info
        return None
    
    def monitor_prices(self, interval: int = 5):
        """Monitor prices for configured symbols
        
        Args:
            interval: Update interval in seconds
        """
        try:
            while True:
                for symbol in self.symbols:
                    price_info = self.client.get_price(symbol)
                    if price_info:
                        bid = price_info.get('bid', 0)
                        ask = price_info.get('ask', 0)
                        logger.info(f"{symbol} - Bid: {bid:.5f}, Ask: {ask:.5f}")
                
                # Check open positions
                positions = self.client.get_positions()
                if positions:
                    logger.info(f"Open Positions: {len(positions)}")
                    for pos in positions:
                        symbol = pos.get('symbol')
                        type_ = pos.get('type')
                        volume = pos.get('volume')
                        profit = pos.get('profit', 0)
                        logger.info(f"Position: {symbol} {type_} {volume} lots, Profit: {profit:.2f}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in price monitoring: {e}")

def main():
    """Main function to run the example trader"""
    trader = ExampleTrader()
    
    # Check account first
    account = trader.check_account()
    if not account:
        logger.error("Failed to connect to MT5 server")
        return
    
    # Start monitoring prices and positions
    logger.info("Starting price monitor...")
    trader.monitor_prices()

if __name__ == "__main__":
    main()