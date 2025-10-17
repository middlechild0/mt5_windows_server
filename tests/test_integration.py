"""
Test suite for trading system integration components.
"""
import pytest
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile

from src.core.prediction_pipeline import PredictionPipeline
from src.core.signal_optimizer import SignalOptimizer
from src.core.risk_manager import RiskProfile
from src.core.model_trainer import ModelTrainer
from src.core.strategy_base import Signal

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h', tz='UTC')
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
        'EURUSD': {
            '1H': data
        }
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def trained_model(sample_data, temp_dir):
    """Create and train a model for testing."""
    trainer = ModelTrainer(
        model_path=f"{temp_dir}/EURUSD_1H_model.joblib",
        scaler_path=f"{temp_dir}/EURUSD_1H_scaler.joblib"
    )
    
    # Prepare and train
    data = sample_data['EURUSD']['1H']
    X, y = trainer.prepare_training_data(data)
    trainer.train(X, y)
    trainer.save_models()
    
    return temp_dir

def test_prediction_pipeline(sample_data, trained_model):
    """Test prediction pipeline functionality."""
    pipeline = PredictionPipeline(models_dir=trained_model)
    pipeline.load_models(['EURUSD'], ['1H'])
    
    # Generate predictions
    timestamp = datetime(2023, 1, 15, tzinfo=UTC)
    signals = pipeline.generate_predictions(sample_data, timestamp)
    
    # Verify predictions
    assert isinstance(signals, list)
    for signal in signals:
        assert signal.pair == 'EURUSD'
        assert signal.timeframe == '1H'
        assert signal.timestamp == timestamp
        assert signal.entry_price > 0
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price
        assert 0 <= signal.confidence <= 1

def test_signal_optimizer():
    """Test signal optimizer functionality."""
    risk_profile = RiskProfile(
        max_risk_per_trade=1.0,
        max_open_trades=3,
        max_daily_loss=5.0,
        max_drawdown=10.0,
        risk_reward_ratio=2.0,
        correlation_limit=0.7
    )
    
    optimizer = SignalOptimizer(risk_profile)
    
    # Create test signals
    signals = [
        Signal(
            pair='EURUSD',
            timeframe='1H',
            direction='buy',
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            timestamp=datetime.now(UTC),
            confidence=0.8
        ),
        Signal(
            pair='GBPUSD',
            timeframe='1H',
            direction='buy',
            entry_price=1.2500,
            stop_loss=1.2450,
            take_profit=1.2600,
            timestamp=datetime.now(UTC),
            confidence=0.7
        )
    ]
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h', tz='UTC')
    data1 = pd.DataFrame({
        'close': np.random.normal(1.1000, 0.0050, len(dates))
    }, index=dates)
    data2 = pd.DataFrame({
        'close': np.random.normal(1.2500, 0.0050, len(dates))
    }, index=dates)
    
    market_data = {
        'EURUSD': {'1H': data1},
        'GBPUSD': {'1H': data2}
    }
    
    # Test optimization with no existing positions
    optimized = optimizer.optimize_signals(
        signals=signals,
        current_positions={},
        account_balance=10000.0,
        market_data=market_data
    )
    
    # Verify optimization results
    assert len(optimized) > 0
    for signal in optimized:
        assert hasattr(signal, 'volume')
        assert signal.volume > 0
        assert signal.confidence >= optimizer.confidence_threshold
        
    # Test with existing positions
    current_positions = {
        'EURUSD': {
            'volume': 1000,
            'risk_amount': 50.0
        }
    }
    
    optimized = optimizer.optimize_signals(
        signals=signals,
        current_positions=current_positions,
        account_balance=10000.0,
        market_data=market_data
    )
    
    # Verify position limits
    assert len(optimized) <= risk_profile.max_open_trades - len(current_positions)