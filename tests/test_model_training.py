"""
Test suite for AI model training components.
"""
import pytest
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile

from src.core.model_trainer import ModelTrainer
from src.core.training_pipeline import TrainingPipeline

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
    
    return data

@pytest.fixture
def temp_dir():
    """Create temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_model_trainer(sample_data, temp_dir):
    """Test basic model training functionality."""
    trainer = ModelTrainer(
        model_path=f"{temp_dir}/test_model.joblib",
        scaler_path=f"{temp_dir}/test_scaler.joblib"
    )
    
    # Prepare and train
    X, y = trainer.prepare_training_data(sample_data)
    metrics = trainer.train(X, y)
    
    # Verify metrics
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['train_accuracy', 'val_accuracy',
                                    'train_precision', 'val_precision'])
    assert all(0 <= v <= 1 for v in metrics.values())
    
    # Test prediction
    features = pd.DataFrame(X, columns=trainer.feature_columns)
    predictions = trainer.predict(features)
    assert len(predictions) == len(X)
    assert all(0 <= p <= 1 for p in predictions)
    
    # Test model saving and loading
    trainer.save_models()
    assert Path(trainer.model_path).exists()
    assert Path(trainer.scaler_path).exists()
    
    new_trainer = ModelTrainer(
        model_path=trainer.model_path,
        scaler_path=trainer.scaler_path
    )
    new_trainer.load_models()
    new_predictions = new_trainer.predict(features)
    np.testing.assert_array_almost_equal(predictions, new_predictions)

def test_training_pipeline(sample_data, temp_dir):
    """Test the full training pipeline."""
    pairs = ['EURUSD', 'GBPUSD']
    timeframes = ['1H', '4H']
    
    pipeline = TrainingPipeline(
        pairs=pairs,
        timeframes=timeframes,
        model_dir=f"{temp_dir}/models",
        data_dir=f"{temp_dir}/data"
    )
    
    # Mock the data handler to return our sample data
    def mock_load_market_data(*args, **kwargs):
        return sample_data
    
    pipeline.data_handler.load_market_data = mock_load_market_data
    
    # Train models
    start_date = datetime(2023, 1, 1, tzinfo=UTC)
    end_date = datetime(2023, 1, 31, tzinfo=UTC)
    metrics = pipeline.train_all_models(start_date, end_date)
    
    # Verify training results
    assert isinstance(metrics, dict)
    for pair in pairs:
        assert pair in metrics
        for tf in timeframes:
            assert tf in metrics[pair]
            assert isinstance(metrics[pair][tf], dict)
            assert 'train_accuracy' in metrics[pair][tf]
            
    # Test model validation
    val_metrics = pipeline.validate_models(
        start_date + timedelta(days=1),
        end_date
    )
    
    # Verify validation results
    assert isinstance(val_metrics, dict)
    for pair in pairs:
        assert pair in val_metrics
        for tf in timeframes:
            assert tf in val_metrics[pair]
            assert isinstance(val_metrics[pair][tf], dict)
            assert 'accuracy' in val_metrics[pair][tf]
            
    # Test cleanup
    pipeline.cleanup_old_models(max_age_days=0)  # Should remove all models
    model_files = list(Path(f"{temp_dir}/models").glob("*.joblib"))
    assert len(model_files) == 0