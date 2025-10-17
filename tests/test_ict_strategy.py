"""
Test suite for ICT Core Strategy and backtesting engine.
"""
import pytest
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path

from src.core.strategy_base import Signal
from src.core.risk_manager import RiskProfile
from src.core.backtest_engine import BacktestEngine, BacktestConfig
from src.strategies.ict_core_strategy import ICTCoreStrategy

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H', tz='UTC')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    
    # Ensure high/low/close are logical
    data['close'] = data['open'] + np.random.normal(0, 0.5, len(dates))
    data['high'] = np.maximum(data['open'], data['close']) + abs(np.random.normal(0, 0.2, len(dates)))
    data['low'] = np.minimum(data['open'], data['close']) - abs(np.random.normal(0, 0.2, len(dates)))
    
    return {
        'XAUUSD': {
            '1H': data
        }
    }

@pytest.fixture
def risk_profile():
    """Create test risk profile."""
    return RiskProfile(
        max_risk_per_trade=1.0,  # 1% risk per trade
        max_open_trades=3,
        max_daily_loss=5.0,  # 5% max daily loss
        max_drawdown=10.0,  # 10% max drawdown
        risk_reward_ratio=2.0,
        correlation_limit=0.7
    )

@pytest.fixture
def backtest_config(risk_profile):
    """Create test backtest configuration."""
    return BacktestConfig(
        initial_balance=10000.0,
        pairs=['XAUUSD'],
        timeframes=['1H'],
        start_date=datetime(2023, 1, 1, tzinfo=UTC),
        end_date=datetime(2023, 1, 31, tzinfo=UTC),
        risk_profile=risk_profile
    )

def test_strategy_signal_generation(sample_data):
    """Test basic signal generation."""
    strategy = ICTCoreStrategy("ICT Test", ['1H'], ['XAUUSD'])
    current_time = datetime(2023, 1, 15, tzinfo=UTC)
    
    signals = strategy.analyze(sample_data['XAUUSD'], current_time)
    
    # We may or may not get signals depending on the random data
    # but they should be properly formed if we do
    for signal in signals:
        assert isinstance(signal, Signal)
        assert signal.pair == 'XAUUSD'
        assert signal.direction in ['buy', 'sell']
        assert signal.entry_price > 0
        assert signal.stop_loss > 0
        assert signal.take_profit > 0
        assert signal.timeframe == '1H'

def test_backtest_execution(sample_data, backtest_config):
    """Test backtest execution."""
    strategy = ICTCoreStrategy("ICT Test", ['1H'], ['XAUUSD'])
    engine = BacktestEngine(backtest_config, strategy)
    
    # Load and run backtest
    engine.load_data(sample_data)
    engine.run()
    
    # Get results
    results = engine.get_results()
    
    # Basic validation of results
    assert 'statistics' in results
    assert 'equity_curve' in results
    assert 'daily_stats' in results
    assert 'final_balance' in results
    assert results['final_balance'] > 0
    
    # Verify equity curve
    equity_curve = results['equity_curve']
    assert not equity_curve.empty
    assert 'timestamp' in equity_curve.columns
    assert 'equity' in equity_curve.columns
    assert 'trade_pnl' in equity_curve.columns

def test_risk_management(sample_data, backtest_config):
    """Test risk management constraints."""
    strategy = ICTCoreStrategy("ICT Test", ['1H'], ['XAUUSD'])
    engine = BacktestEngine(backtest_config, strategy)
    engine.load_data(sample_data)
    engine.run()
    
    results = engine.get_results()
    
    # Verify risk management
    trades = results['trades']
    
    # Check position sizes
    for trade in trades:
        # Calculate trade risk
        risk = abs(trade['entry_price'] - trade['stop_loss']) * trade['volume']
        max_risk = backtest_config.initial_balance * (backtest_config.risk_profile.max_risk_per_trade / 100)
        assert risk <= max_risk * 1.01  # Allow 1% margin for rounding
    
    # Check maximum open trades
    max_concurrent = 0
    if trades:
        timestamps = [trade['entry_time'] for trade in trades] + [trade['exit_time'] for trade in trades if trade.get('exit_time')]
        unique_timestamps = sorted(set(timestamps))
        current_open = 0
        
        for t in unique_timestamps:
            opens = sum(1 for trade in trades if trade['entry_time'] == t)
            closes = sum(1 for trade in trades if trade.get('exit_time') == t)
            current_open = current_open + opens - closes
            max_concurrent = max(max_concurrent, current_open)
    
    assert max_concurrent <= backtest_config.risk_profile.max_open_trades