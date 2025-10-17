"""
Test suite for MT5 connector.
"""
import pytest
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.core.mt5_connector import MT5Connector, MT5Config

# Mock MetaTrader5 module
mt5_mock = MagicMock()

@pytest.fixture
def mt5_config():
    """Create sample MT5 configuration."""
    return MT5Config(
        path="C:\\Program Files\\MetaTrader 5",
        login=12345,
        password="password",
        server="Demo Server",
        timeout=10000
    )

@patch('src.core.mt5_connector.mt5', mt5_mock)
def test_mt5_connection(mt5_config):
    """Test MT5 connection functionality."""
    # Setup mock
    mt5_mock.initialize.return_value = True
    mt5_mock.last_error.return_value = ""
    
    # Test connection
    connector = MT5Connector(mt5_config)
    assert connector.connect()
    assert connector.connected
    
    # Test disconnection
    connector.disconnect()
    assert not connector.connected
    mt5_mock.shutdown.assert_called_once()

@patch('src.core.mt5_connector.mt5', mt5_mock)
def test_get_real_time_data(mt5_config):
    """Test getting real-time data."""
    # Setup mock data
    sample_bars = [
        {
            'time': int(datetime.now().timestamp()),
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'tick_volume': 100,
            'spread': 2,
            'real_volume': 10000
        }
    ]
    mt5_mock.copy_rates_from_pos.return_value = sample_bars
    mt5_mock.initialize.return_value = True
    
    # Test data retrieval
    connector = MT5Connector(mt5_config)
    connector.connect()
    
    df = connector.get_real_time_data('EURUSD', '1h', 1)
    assert df is not None
    assert not df.empty
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns

@patch('src.core.mt5_connector.mt5', mt5_mock)
def test_place_market_order(mt5_config):
    """Test placing market orders."""
    # Setup mock
    class OrderResult:
        def __init__(self):
            self.order = 12345
            self.retcode = mt5_mock.TRADE_RETCODE_DONE
            self.comment = "Success"
    
    mt5_mock.TRADE_RETCODE_DONE = 10009
    mt5_mock.order_send.return_value = OrderResult()
    mt5_mock.initialize.return_value = True
    mt5_mock.symbol_info_tick.return_value = MagicMock(ask=1.1000, bid=1.0998)
    
    # Test order placement
    connector = MT5Connector(mt5_config)
    connector.connect()
    
    success, order_id = connector.place_market_order(
        symbol='EURUSD',
        order_type='buy',
        volume=0.1,
        sl=1.0950,
        tp=1.1050
    )
    
    assert success
    assert order_id == '12345'
    assert mt5_mock.order_send.called

@patch('src.core.mt5_connector.mt5', mt5_mock)
def test_get_account_info(mt5_config):
    """Test getting account information."""
    # Setup mock
    class AccountInfo:
        def _asdict(self):
            return {
                'balance': 10000.0,
                'equity': 10100.0,
                'margin': 100.0,
                'margin_free': 9900.0,
                'leverage': 100,
                'currency': 'USD'
            }
    
    mt5_mock.account_info.return_value = AccountInfo()
    mt5_mock.initialize.return_value = True
    
    # Test account info retrieval
    connector = MT5Connector(mt5_config)
    connector.connect()
    
    info = connector.get_account_info()
    assert info['balance'] == 10000.0
    assert info['equity'] == 10100.0
    assert info['margin'] == 100.0
    assert info['free_margin'] == 9900.0
    assert info['leverage'] == 100
    assert info['currency'] == 'USD'