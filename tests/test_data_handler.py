"""
Test suite for data handler functionality.
"""
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import shutil
import sqlite3
from datetime import datetime, timedelta

# Add project root to path for imports
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.data_handler import DataHandler
from src.config import PROJECT_CONFIG

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with clean database."""
    # Create test directories
    test_db_dir = Path(PROJECT_CONFIG['database']['path']).parent
    test_db_dir.mkdir(parents=True, exist_ok=True)
    
    backup_dir = Path(PROJECT_CONFIG['database']['backup_dir'])
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up after tests
    yield
    
    # Remove test database and backup files
    shutil.rmtree(test_db_dir, ignore_errors=True)
    shutil.rmtree(backup_dir, ignore_errors=True)

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='1min', tz='UTC')
    data = pd.DataFrame({
        'time': dates,
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    })
    return data

@pytest.fixture
def temp_csv(sample_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

def test_data_loading(temp_csv):
    """Test loading data from CSV."""
    handler = DataHandler()
    df = handler.load_raw_csv(temp_csv, "XAUUSD", "1M")
    
    assert not df.empty, "DataFrame should not be empty"
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert df.index.tz is not None, "Index should have timezone info"
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
        "All OHLCV columns should be present"

def test_data_normalization(temp_csv):
    """Test data normalization process."""
    handler = DataHandler()
    df = handler.load_raw_csv(temp_csv, "XAUUSD", "1M")
    df_norm = handler.normalize_dataframe(df)
    
    assert 'atr_14' in df_norm.columns, "ATR column should be present"
    assert 'volatility' in df_norm.columns, "Volatility column should be present"
    assert 'volume_ma' in df_norm.columns, "Volume MA column should be present"
    assert not df_norm.isnull().any().any(), "No null values should be present"

def test_data_saving(temp_csv):
    """Test saving processed data."""
    handler = DataHandler()
    df = handler.load_raw_csv(temp_csv, "XAUUSD", "1M")
    df_norm = handler.normalize_dataframe(df)
    saved_path = handler.save_clean_data(df_norm, "XAUUSD", "1M")
    
    assert Path(saved_path).exists(), "Saved file should exist"
    assert saved_path.endswith('.parquet'), "Should save as parquet file"
    
    # Verify database entry
    with sqlite3.connect(handler.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pairs WHERE pair = ?", ("XAUUSD",))
        row = cursor.fetchone()
        assert row is not None, "Database entry should exist"
        assert row[3] == "1M", "Timeframe should match"

def test_validation_logging(temp_csv):
    """Test validation logging functionality."""
    handler = DataHandler()
    df = handler.load_raw_csv(temp_csv, "XAUUSD", "1M")
    
    # Check validation logs
    with sqlite3.connect(handler.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM data_validation 
            WHERE pair = ? AND validation_type = 'data_load'
        """, ("XAUUSD",))
        row = cursor.fetchone()
        assert row is not None, "Validation log should exist"
        assert row[4] == 'success', "Validation should be successful"

def test_error_handling():
    """Test error handling for invalid data."""
    handler = DataHandler()
    
    with pytest.raises(Exception):
        # Try to load non-existent file
        handler.load_raw_csv("nonexistent.csv", "XAUUSD", "1M")
    
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        # Create invalid CSV (missing required columns)
        f.write("timestamp,value\n2023-01-01,1.0\n")
        f.flush()
        
        with pytest.raises(Exception):
            handler.load_raw_csv(f.name, "XAUUSD", "1M")
    
    os.unlink(f.name)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])