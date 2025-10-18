"""
Real-time prediction pipeline for integrating AI models with trading system.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

from src.core.model_trainer import ModelTrainer
from src.core.strategy_base import Signal
from src.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self,
                 models_dir: str = "models",
                 min_prediction_threshold: float = 0.65):
        """
        Initialize prediction pipeline.
        
        Args:
            models_dir: Directory containing trained models
            min_prediction_threshold: Minimum probability threshold for generating signals
        """
        self.models_dir = Path(models_dir)
        self.min_prediction_threshold = min_prediction_threshold
        self.models: Dict[str, Dict[str, ModelTrainer]] = {}
        self.feature_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
    def load_models(self, pairs: List[str], timeframes: List[str]) -> None:
        """
        Load trained models for specified pairs and timeframes.
        
        Args:
            pairs: List of currency pairs
            timeframes: List of timeframes
        """
        for pair in pairs:
            self.models[pair] = {}
            self.feature_cache[pair] = {}
            
            for tf in timeframes:
                model_path = self.models_dir / f"{pair}_{tf}_model.joblib"
                scaler_path = self.models_dir / f"{pair}_{tf}_scaler.joblib"
                
                if not model_path.exists() or not scaler_path.exists():
                    logger.warning(f"No model found for {pair} {tf}")
                    continue
                
                try:
                    trainer = ModelTrainer(
                        model_path=str(model_path),
                        scaler_path=str(scaler_path)
                    )
                    trainer.load_models()
                    self.models[pair][tf] = trainer
                    logger.info(f"Loaded model for {pair} {tf}")
                except Exception as e:
                    logger.error(f"Failed to load model for {pair} {tf}: {e}")
    
    def generate_predictions(self,
                           market_data: Dict[str, Dict[str, pd.DataFrame]],
                           timestamp: datetime) -> List[Signal]:
        """
        Generate trading signals using loaded models.
        
        Args:
            market_data: Dictionary of OHLCV data for each pair/timeframe
            timestamp: Current timestamp
            
        Returns:
            List of generated trading signals
        """
        signals = []
        
        for pair, timeframes in self.models.items():
            if pair not in market_data:
                continue
                
            for tf, trainer in timeframes.items():
                if tf not in market_data[pair]:
                    continue
                
                try:
                    # Get latest data
                    data = market_data[pair][tf]
                    if data.empty:
                        continue
                    
                    # Prepare features
                    X, _ = trainer.prepare_training_data(data)
                    features = pd.DataFrame(X, columns=trainer.feature_columns)
                    
                    # Generate prediction
                    prob = trainer.predict(features)[-1]  # Get latest prediction
                    
                    # Generate signal if probability exceeds threshold
                    if prob > self.min_prediction_threshold:
                        signal = self._create_signal(
                            pair=pair,
                            timeframe=tf,
                            data=data,
                            probability=prob,
                            timestamp=timestamp
                        )
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating prediction for {pair} {tf}: {e}")
        
        return signals
    
    def _create_signal(self,
                      pair: str,
                      timeframe: str,
                      data: pd.DataFrame,
                      probability: float,
                      timestamp: datetime) -> Signal:
        """
        Create a trading signal from model prediction.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            data: Market data
            probability: Model prediction probability
            timestamp: Current timestamp
            
        Returns:
            Trading signal
        """
        # Get latest prices
        latest = data.iloc[-1]
        close = latest['close']
        
        # Calculate ATR for dynamic stop loss
        atr = self._calculate_atr(data, period=14)
        
        # Set stop loss and take profit based on ATR
        stop_distance = atr * 1.5
        take_distance = atr * 3.0  # 2:1 reward-risk ratio
        
        return Signal(
            pair=pair,
            timeframe=timeframe,
            direction='buy',  # We're predicting profitable trades only
            entry_price=close,
            stop_loss=close - stop_distance,
            take_profit=close + take_distance,
            timestamp=timestamp,
            confidence=probability
        )
    
    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return float(atr)