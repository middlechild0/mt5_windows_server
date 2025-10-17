"""
Data handling utilities for market data loading and preprocessing.
"""
from typing import Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data handler.
        
        Args:
            data_dir: Base directory for market data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_market_data(self,
                        pair: str,
                        timeframe: str,
                        start_date: datetime,
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load market data for a given pair and timeframe.
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Data timeframe (e.g., '1H', '4H')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        # Try processed data first
        processed_file = self.processed_dir / f"{pair}_{timeframe}.parquet"
        if processed_file.exists():
            try:
                df = pd.read_parquet(processed_file)
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return df
            except Exception as e:
                logger.error(f"Error loading processed data: {e}")
        
        # Fall back to raw data
        raw_file = self.raw_dir / f"{pair}_{timeframe}.parquet"
        if raw_file.exists():
            try:
                df = pd.read_parquet(raw_file)
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return self._preprocess_data(df)
            except Exception as e:
                logger.error(f"Error loading raw data: {e}")
        
        logger.warning(f"No data found for {pair} {timeframe}")
        return None
    
    def save_market_data(self,
                        data: pd.DataFrame,
                        pair: str,
                        timeframe: str,
                        is_processed: bool = False) -> None:
        """
        Save market data to parquet file.
        
        Args:
            data: DataFrame with OHLCV data
            pair: Currency pair
            timeframe: Data timeframe
            is_processed: Whether this is processed data
        """
        output_dir = self.processed_dir if is_processed else self.raw_dir
        output_file = output_dir / f"{pair}_{timeframe}.parquet"
        
        try:
            data.to_parquet(output_file)
            logger.info(f"Saved {pair} {timeframe} data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw market data.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed data
        """
        df = data.copy()
        
        # Ensure proper column names
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Missing required columns. Expected {expected_cols}")
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df = df.sort_index()
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add basic features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def get_available_pairs(self) -> Dict[str, list]:
        """
        Get available pairs and timeframes from data directory.
        
        Returns:
            Dictionary mapping pairs to available timeframes
        """
        pairs = {}
        for file in self.raw_dir.glob("*.parquet"):
            try:
                pair, timeframe = file.stem.split("_")
                if pair not in pairs:
                    pairs[pair] = []
                pairs[pair].append(timeframe)
            except ValueError:
                continue
        return pairs