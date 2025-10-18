"""
Model trainer for the ICT AI trading system.
Uses lightweight ML models that can be run locally.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, 
                 model_path: str = "models/ict_model.joblib",
                 scaler_path: str = "models/scaler.joblib"):
        """Initialize model trainer with paths for saving artifacts."""
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.feature_columns = []
        
    def prepare_training_data(self, 
                            data: pd.DataFrame, 
                            lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by engineering features and creating labels.
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods to look back for feature engineering
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        df = data.copy()
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(lookback).std()
        df['rsi'] = self._calculate_rsi(df['close'], periods=14)
        df['ma_fast'] = df['close'].rolling(10).mean()
        df['ma_slow'] = df['close'].rolling(30).mean()
        df['ma_trend'] = (df['ma_fast'] > df['ma_slow']).astype(int)
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(lookback).mean()
        df['volume_trend'] = (df['volume'] > df['volume_ma']).astype(int)
        
        # Create labels (1 for profitable trades, 0 for unprofitable)
        df['forward_returns'] = df['close'].pct_change(5).shift(-5)
        df['label'] = (df['forward_returns'] > 0).astype(int)
        
        # Select features
        self.feature_columns = [
            'returns', 'volatility', 'rsi', 'ma_trend',
            'body_size', 'upper_wick', 'lower_wick',
            'volume_trend'
        ]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        X = df[self.feature_columns].values
        y = df['label'].values
        
        return X, y
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              test_size: float = 0.2) -> Dict:
        """
        Train the model using gradient boosting classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Initialize models
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': []
        }
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            metrics['train_accuracy'].append(
                np.mean(train_pred == y_train)
            )
            metrics['val_accuracy'].append(
                np.mean(val_pred == y_val)
            )
            metrics['train_precision'].append(
                np.sum((train_pred == 1) & (y_train == 1)) / np.sum(train_pred == 1)
            )
            metrics['val_precision'].append(
                np.sum((val_pred == 1) & (y_val == 1)) / np.sum(val_pred == 1)
            )
        
        # Save models
        self.save_models()
        
        # Return average metrics
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
            
        X = features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_models(self) -> None:
        """Save the trained model and scaler."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Saved model to {self.model_path}")
    
    def load_models(self) -> None:
        """Load saved model and scaler."""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Set feature columns from model
        self.feature_columns = [
            'returns', 'volatility', 'rsi', 'ma_trend',
            'body_size', 'upper_wick', 'lower_wick',
            'volume_trend'
        ]
        logger.info(f"Loaded model from {self.model_path}")
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))