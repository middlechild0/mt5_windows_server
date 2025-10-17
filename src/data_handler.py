"""
Enhanced data handler with comprehensive validation, logging, and backup features.
Handles all data operations in alignment with the project's organized structure.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure datetime adapters for SQLite
def adapt_datetime(dt):
    """Convert datetime to UTC ISO format string."""
    return dt.astimezone(UTC).isoformat()

def convert_datetime(val):
    """Convert ISO format string to UTC datetime."""
    return datetime.fromisoformat(val)

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)
import logging
from typing import Optional, Dict, Union, List, Tuple
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))
from .config import PROJECT_CONFIG
from .utils.file_utils import create_backup, validate_directory_structure, archive_old_data

# Configure logging
def setup_logging():
    """Setup logging with file and console handlers."""
    log_dir = Path(PROJECT_CONFIG['log_dirs']['system'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'data_handler.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass

class DataHandler:
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DataHandler with enhanced error checking and logging.
        
        Args:
            db_path: Optional database path, defaults to config value
        """
        self.db_path = Path(db_path or PROJECT_CONFIG['database']['path'])
        self.backup_dir = Path(PROJECT_CONFIG['database']['backup_dir'])
        
        # Validate directory structure
        self._validate_environment()
        
        # Initialize database
        self._init_database()
        
        logger.info("DataHandler initialized successfully")

    def _validate_environment(self) -> None:
        """Validate and create necessary directories."""
        try:
            validate_directory_structure(PROJECT_CONFIG)
            logger.info("Directory structure validated successfully")
        except Exception as e:
            logger.error(f"Failed to validate directory structure: {e}")
            raise

    def _init_database(self) -> None:
        """Initialize database with enhanced schema and safety checks."""
        try:
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup of existing database if it exists
            if self.db_path.exists():
                backup_path = create_backup(
                    str(self.db_path),
                    str(self.backup_dir)
                )
                logger.info(f"Created database backup at {backup_path}")

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables with more detailed schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pairs (
                        pair TEXT PRIMARY KEY,
                        last_update TIMESTAMP,
                        data_path TEXT,
                        timeframe TEXT,
                        first_timestamp TIMESTAMP,
                        last_timestamp TIMESTAMP,
                        row_count INTEGER,
                        status TEXT CHECK(status IN ('active', 'archived', 'invalid'))
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_validation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT,
                        timestamp TIMESTAMP,
                        validation_type TEXT,
                        status TEXT,
                        details TEXT,
                        FOREIGN KEY(pair) REFERENCES pairs(pair)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def load_raw_csv(self, path: str, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Enhanced CSV loader with comprehensive validation and logging.
        
        Args:
            path: Path to CSV file
            pair: Trading pair identifier
            timeframe: Data timeframe (e.g., "1M", "5M", "1H")
            
        Returns:
            Validated DataFrame with UTC datetime index
        """
        try:
            logger.info(f"Loading CSV for {pair} ({timeframe}) from {path}")
            
            # Read CSV
            df = pd.read_csv(path)
            
            # Validate structure
            self._validate_dataframe_structure(df)
            
            # Process timestamps
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            # Validate continuity
            gaps = self._check_timestamp_continuity(df, timeframe)
            if gaps:
                logger.warning(f"Found {len(gaps)} time gaps in {pair} data")
                self._log_validation_issue(pair, "timestamp_gaps", gaps)
            
            # Record validation in database
            self._log_validation_success(pair, "data_load")
            
            logger.info(f"Successfully loaded {len(df)} rows for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV for {pair}: {e}")
            self._log_validation_issue(pair, "data_load_failed", str(e))
            raise

    def _validate_dataframe_structure(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns and data types."""
        required_columns = {'time', 'open', 'high', 'low', 'close', 'volume'}
        missing_cols = required_columns - set(df.columns)
        
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df[list(required_columns)].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")

    def _check_timestamp_continuity(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Check for gaps in timestamp sequence."""
        timeframe_minutes = {
            "1M": 1,
            "5M": 5,
            "15M": 15,
            "1H": 60,
            "4H": 240,
            "1D": 1440
        }
        
        expected_delta = pd.Timedelta(minutes=timeframe_minutes[timeframe])
        time_diff = df.index.to_series().diff()
        gaps = time_diff[time_diff > expected_delta]
        
        return [
            {
                "start": str(idx - expected_delta),
                "end": str(idx),
                "gap_size": str(gap)
            }
            for idx, gap in gaps.items()
        ]

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced normalization with additional technical indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Normalized DataFrame with additional columns
        """
        try:
            logger.info("Starting data normalization")
            
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Ensure float dtypes
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Compute ATR(14)
            df['atr_14'] = self._compute_atr(df, period=14)
            
            # Add volatility measures
            df['volatility'] = df['high'] - df['low']
            df['volatility_ma'] = df['volatility'].rolling(window=14, min_periods=1).mean()
            
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_std'] = df['volume'].rolling(window=20, min_periods=1).std()
            
            # Forward fill any remaining NaNs
            df = df.ffill().bfill()
            
            remaining_nulls = df.isnull().sum()
            if remaining_nulls.any():
                logger.warning(f"Remaining null values after normalization:\n{remaining_nulls[remaining_nulls > 0]}")
            
            logger.info("Data normalization completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error during normalization: {e}")
            raise

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()

    def save_clean_data(self, df: pd.DataFrame, pair: str, timeframe: str) -> str:
        """
        Enhanced data saving with validation and backup.
        
        Args:
            df: Processed DataFrame to save
            pair: Trading pair identifier
            timeframe: Data timeframe
            
        Returns:
            Path where data was saved
        """
        try:
            # Create pair-specific directory
            data_dir = Path(PROJECT_CONFIG['data_dirs']['processed']) / pair.lower() / timeframe
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = data_dir / f"{pair.lower()}_{timeframe}_{timestamp}.parquet"
            
            # Create backup if file exists
            if data_path.exists():
                create_backup(str(data_path), str(PROJECT_CONFIG['data_dirs']['archive']))
            
            # Save data
            df.to_parquet(data_path)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO pairs 
                    (pair, last_update, data_path, timeframe, first_timestamp, 
                     last_timestamp, row_count, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair,
                    datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    str(data_path),
                    timeframe,
                    df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    df.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                    len(df),
                    'active'
                ))
                conn.commit()
            
            logger.info(f"Successfully saved clean data for {pair} to {data_path}")
            return str(data_path)
            
        except Exception as e:
            logger.error(f"Error saving clean data for {pair}: {e}")
            raise

    def _log_validation_issue(self, pair: str, validation_type: str, details: Union[str, List, Dict]) -> None:
        """Log validation issues to database."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_validation 
                (pair, timestamp, validation_type, status, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pair,
                datetime.now(UTC),
                validation_type,
                'failed',
                str(details)
            ))
            conn.commit()

    def _log_validation_success(self, pair: str, validation_type: str) -> None:
        """Log successful validation to database."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_validation 
                (pair, timestamp, validation_type, status, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pair,
                datetime.now(UTC),
                validation_type,
                'success',
                'Validation passed'
            ))
            conn.commit()

def test_roundtrip():
    """
    Unit test to verify data roundtrip integrity
    """
    import tempfile
    import os
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='1min', tz='UTC')
    test_data = pd.DataFrame({
        'time': dates,
        'open': np.random.rand(len(dates)),
        'high': np.random.rand(len(dates)),
        'low': np.random.rand(len(dates)),
        'close': np.random.rand(len(dates)),
        'volume': np.random.rand(len(dates))
    })
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        
    try:
        # Initialize handler
        handler = DataHandler()
        
        # Load and process data
        df = handler.load_raw_csv(f.name)
        df = handler.normalize_dataframe(df)
        
        # Verify data integrity
        assert len(df) == len(test_data), "Row count mismatch"
        assert 'atr_14' in df.columns, "ATR column missing"
        assert df.index.tzinfo is not None, "Timezone information missing"
        
        print("Roundtrip test passed successfully!")
        
    finally:
        os.unlink(f.name)

if __name__ == "__main__":
    # Run unit test
    test_roundtrip()