"""
Script to test Exness MT5 connection and data streaming.
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timezone
import time

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.mt5_connector import MT5Connector, MT5Config
from src.core.broker_config import ExnessBrokerConfig
from src.core.market_data_stream import MarketDataStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mt5_connection(login: int, password: str, path: str = "") -> None:
    """
    Test MT5 connection with Exness.
    
    Args:
        login: MT5 account number
        password: MT5 account password
        path: Path to MT5 terminal (optional)
    """
    # Initialize configuration
    broker_config = ExnessBrokerConfig()
    mt5_config = MT5Config(
        path=path,
        login=login,
        password=password,
        server=broker_config.server_demo,  # Using demo server
        timeout=10000
    )
    
    # Create MT5 connector
    connector = MT5Connector(mt5_config)
    
    try:
        # Connect to MT5
        logger.info("Connecting to Exness MT5...")
        if not connector.connect():
            logger.error("Failed to connect to MT5")
            return
            
        # Get account info
        account_info = connector.get_account_info()
        logger.info(f"Account Info:")
        logger.info(f"  Balance: {account_info['balance']:.2f} {account_info['currency']}")
        logger.info(f"  Equity: {account_info['equity']:.2f} {account_info['currency']}")
        logger.info(f"  Free Margin: {account_info['free_margin']:.2f} {account_info['currency']}")
        logger.info(f"  Leverage: 1:{account_info['leverage']}")
        
        # Initialize market data stream
        logger.info("\nInitializing market data stream...")
        stream = MarketDataStream(broker_config.common_pairs[:3], connector)  # Test with first 3 pairs
        
        def on_tick(update):
            logger.info(f"Tick: {update.pair} Bid: {update.bid:.5f} Ask: {update.ask:.5f}")
        
        # Register callback and start stream
        stream.register_callback(on_tick)
        stream.start()
        
        logger.info("\nStreaming data for 30 seconds...")
        time.sleep(30)
        
        # Stop stream
        stream.stop()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        
    finally:
        # Cleanup
        connector.disconnect()
        logger.info("Test completed")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_exness.py <login> <password> [mt5_path]")
        sys.exit(1)
        
    login = int(sys.argv[1])
    password = sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else ""
    
    test_mt5_connection(login, password, path)