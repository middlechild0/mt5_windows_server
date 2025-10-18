"""
Pipeline for training and maintaining AI models.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from src.core.model_trainer import ModelTrainer
from src.core.data_handler import DataHandler
from src.utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self,
                 pairs: List[str],
                 timeframes: List[str],
                 model_dir: str = "models",
                 data_dir: str = "data"):
        """
        Initialize the training pipeline.
        
        Args:
            pairs: List of currency pairs to train on
            timeframes: List of timeframes to use
            model_dir: Directory to store trained models
            data_dir: Directory containing market data
        """
        self.pairs = pairs
        self.timeframes = timeframes
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Ensure directories exist
        ensure_dir(self.model_dir)
        ensure_dir(self.data_dir)
        
        # Initialize components
        self.data_handler = DataHandler()
        self.trainers = {}
        
        # Create model trainers for each pair/timeframe
        for pair in pairs:
            self.trainers[pair] = {}
            for tf in timeframes:
                model_path = self.model_dir / f"{pair}_{tf}_model.joblib"
                scaler_path = self.model_dir / f"{pair}_{tf}_scaler.joblib"
                self.trainers[pair][tf] = ModelTrainer(
                    model_path=str(model_path),
                    scaler_path=str(scaler_path)
                )
    
    def train_all_models(self,
                        start_date: datetime,
                        end_date: datetime,
                        force_retrain: bool = False) -> Dict:
        """
        Train models for all pairs and timeframes.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            force_retrain: Whether to force retraining even if model exists
            
        Returns:
            Dictionary with training metrics for each model
        """
        metrics = {}
        
        for pair in self.pairs:
            metrics[pair] = {}
            for tf in self.timeframes:
                logger.info(f"Training model for {pair} {tf}")
                
                # Check if model exists and we're not force retraining
                trainer = self.trainers[pair][tf]
                if not force_retrain and trainer.model_path.exists():
                    try:
                        trainer.load_models()
                        logger.info(f"Loaded existing model for {pair} {tf}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load model for {pair} {tf}: {e}")
                
                # Load and prepare data
                data = self.data_handler.load_market_data(
                    pair, tf, start_date, end_date
                )
                
                if data is None or data.empty:
                    logger.warning(f"No data available for {pair} {tf}")
                    continue
                
                # Prepare training data
                X, y = trainer.prepare_training_data(data)
                
                # Train model
                try:
                    model_metrics = trainer.train(X, y)
                    metrics[pair][tf] = model_metrics
                    logger.info(f"Successfully trained model for {pair} {tf}")
                except Exception as e:
                    logger.error(f"Failed to train model for {pair} {tf}: {e}")
                    metrics[pair][tf] = {"error": str(e)}
        
        return metrics
    
    def validate_models(self,
                       start_date: datetime,
                       end_date: datetime) -> Dict:
        """
        Validate all trained models on recent data.
        
        Args:
            start_date: Start date for validation data
            end_date: End date for validation data
            
        Returns:
            Dictionary with validation metrics for each model
        """
        metrics = {}
        
        for pair in self.pairs:
            metrics[pair] = {}
            for tf in self.timeframes:
                logger.info(f"Validating model for {pair} {tf}")
                
                trainer = self.trainers[pair][tf]
                if not trainer.model_path.exists():
                    logger.warning(f"No model found for {pair} {tf}")
                    continue
                
                try:
                    # Load model if not already loaded
                    if trainer.model is None:
                        trainer.load_models()
                    
                    # Load validation data
                    data = self.data_handler.load_market_data(
                        pair, tf, start_date, end_date
                    )
                    
                    if data is None or data.empty:
                        logger.warning(f"No validation data available for {pair} {tf}")
                        continue
                    
                    # Prepare validation data
                    X, y = trainer.prepare_training_data(data)
                    
                    # Generate predictions
                    predictions = trainer.predict(pd.DataFrame(X, columns=trainer.feature_columns))
                    
                    # Calculate metrics
                    thresh_predictions = (predictions > 0.5).astype(int)
                    accuracy = (thresh_predictions == y).mean()
                    precision = ((thresh_predictions == 1) & (y == 1)).sum() / thresh_predictions.sum()
                    
                    metrics[pair][tf] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision)
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to validate model for {pair} {tf}: {e}")
                    metrics[pair][tf] = {"error": str(e)}
        
        return metrics
    
    def cleanup_old_models(self, max_age_days: int = 30) -> None:
        """Remove model files older than specified age."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for model_file in self.model_dir.glob("*.joblib"):
            if model_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    model_file.unlink()
                    logger.info(f"Removed old model file: {model_file}")
                except Exception as e:
                    logger.error(f"Failed to remove old model file {model_file}: {e}")