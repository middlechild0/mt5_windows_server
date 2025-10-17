"""
Test suite for live trading components.
"""
import pytest
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from typing import List

from src.core.market_data_stream import MarketDataStream, MarketUpdate
from src.core.live_trade_executor import (
    LiveTradeExecutor, Order, Position, OrderType, OrderStatus
)
from src.core.strategy_base import Signal

def test_market_data_stream():
    """Test market data streaming functionality."""
    pairs = ['EURUSD', 'GBPUSD']
    stream = MarketDataStream(pairs)
    
    updates: List[MarketUpdate] = []
    def callback(update: MarketUpdate):
        updates.append(update)
    
    # Register callback and start stream
    stream.register_callback(callback)
    stream.start()
    
    # Process some updates
    now = datetime.now(UTC)
    update1 = MarketUpdate(
        pair='EURUSD',
        timestamp=now,
        bid=1.1000,
        ask=1.1001,
        volume=100000
    )
    update2 = MarketUpdate(
        pair='EURUSD',
        timestamp=now + timedelta(seconds=30),
        bid=1.1002,
        ask=1.1003,
        volume=150000
    )
    
    stream.process_update(update1)
    stream.process_update(update2)
    
    # Verify callback was called
    assert len(updates) == 2
    assert updates[0].pair == 'EURUSD'
    assert updates[0].bid == 1.1000
    
    # Check OHLCV data generation
    ohlcv = stream.get_latest_data('EURUSD', '1m')
    assert ohlcv is not None
    assert not ohlcv.empty
    assert len(ohlcv) == 1  # Both updates within same minute
    
    latest_candle = ohlcv.iloc[-1]
    assert latest_candle['open'] == pytest.approx((1.1000 + 1.1001) / 2)
    assert latest_candle['close'] == pytest.approx((1.1002 + 1.1003) / 2)
    assert latest_candle['volume'] == 250000
    
    # Test stream stop
    stream.stop()
    update3 = MarketUpdate(
        pair='EURUSD',
        timestamp=now + timedelta(seconds=60),
        bid=1.1004,
        ask=1.1005,
        volume=200000
    )
    stream.process_update(update3)
    assert len(updates) == 2  # No new updates after stop

def test_live_trade_executor():
    """Test live trade execution functionality."""
    account_config = {
        'initial_balance': 10000.0,
        'leverage': 100
    }
    
    executor = LiveTradeExecutor(account_config)
    
    # Create a test signal
    signal = Signal(
        pair='EURUSD',
        timeframe='1H',
        direction='buy',
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        timestamp=datetime.now(UTC),
        confidence=0.8,
        volume=100000.0
    )
    
    # Place market order
    success, order_id = executor.place_market_order(signal, 1.1000)
    assert success
    assert order_id is not None
    
    # Verify order was processed
    assert len(executor.orders) == 1
    assert len(executor.positions) == 1
    
    order = executor.orders[order_id]
    assert order.status == OrderStatus.FILLED
    assert order.pair == 'EURUSD'
    assert order.direction == 'buy'
    assert order.volume == 100000.0
    
    # Test position updates with market data
    position = executor.get_position('EURUSD')
    assert position is not None
    assert position.entry_price == 1.1000
    
    # Update with profit
    update = MarketUpdate(
        pair='EURUSD',
        timestamp=datetime.now(UTC),
        bid=1.1050,
        ask=1.1051,
        volume=100000
    )
    closed_positions = executor.update_positions(update)
    assert len(closed_positions) == 0  # Position still open
    assert position.unrealized_pnl == pytest.approx(500.0)  # (1.1050 - 1.1000) * 100000
    
    # Update to trigger stop loss
    update = MarketUpdate(
        pair='EURUSD',
        timestamp=datetime.now(UTC),
        bid=1.0949,
        ask=1.0950,
        volume=100000
    )
    closed_positions = executor.update_positions(update)
    assert len(closed_positions) == 1  # Position closed
    assert len(executor.positions) == 0
    
    # Verify equity curve
    assert len(executor.equity_curve) > 0
    latest_equity = executor.equity_curve[-1]
    assert latest_equity['balance'] < account_config['initial_balance']  # Loss due to stop loss