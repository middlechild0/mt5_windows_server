"""
MT5 client for communicating with Windows MT5 server.
"""
import requests
import logging
from typing import Dict, Optional, List
import json
from datetime import datetime

class MT5Client:
    """Client for interacting with MT5 server running on Windows"""
    
    def __init__(self, server_ip: str, port: int = 5000):
        """
        Initialize MT5 client.
        
        Args:
            server_ip: IP address of Windows machine running MT5 server
            port: Port number (default 5000)
        """
        self.base_url = f"http://{server_ip}:{port}"
        self.logger = logging.getLogger(__name__)
        
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            response = requests.get(f"{self.base_url}/account_info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
            
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol"""
        try:
            response = requests.get(f"{self.base_url}/price/{symbol}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
            
    def get_positions(self) -> Optional[List[Dict]]:
        """Get all open positions"""
        try:
            response = requests.get(f"{self.base_url}/positions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return None

def test_connection(server_ip: str):
    """Test connection to MT5 server"""
    client = MT5Client(server_ip)
    
    print("\nTesting MT5 Server Connection...")
    print("=" * 50)
    
    # Test account info
    print("\nGetting Account Info:")
    account = client.get_account_info()
    if account:
        print(json.dumps(account, indent=2))
    else:
        print("Failed to get account info")
    
    # Test price data
    print("\nGetting EURUSD Price:")
    price = client.get_price("EURUSD")
    if price:
        print(json.dumps(price, indent=2))
    else:
        print("Failed to get EURUSD price")
    
    # Test positions
    print("\nGetting Open Positions:")
    positions = client.get_positions()
    if positions:
        print(json.dumps(positions, indent=2))
    else:
        print("Failed to get positions")

if __name__ == "__main__":
    # Replace with your Windows machine's IP address
    WINDOWS_IP = "192.168.1.xxx"  # Replace with actual IP
    test_connection(WINDOWS_IP)