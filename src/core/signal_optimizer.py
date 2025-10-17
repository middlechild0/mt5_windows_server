"""
Risk adjustment system for AI-generated signals.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.core.strategy_base import Signal
from src.core.risk_manager import RiskProfile

logger = logging.getLogger(__name__)

class SignalOptimizer:
    def __init__(self, risk_profile: RiskProfile):
        """
        Initialize signal optimizer.
        
        Args:
            risk_profile: Risk management parameters
        """
        self.risk_profile = risk_profile
        self.position_sizing_factor = 1.0
        self.confidence_threshold = 0.65
        
    def optimize_signals(self,
                        signals: List[Signal],
                        current_positions: Dict[str, Dict],
                        account_balance: float,
                        market_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Signal]:
        """
        Optimize and filter trading signals based on risk parameters.
        
        Args:
            signals: List of raw trading signals
            current_positions: Currently open positions
            account_balance: Current account balance
            market_data: Market data for correlation analysis
            
        Returns:
            List of optimized trading signals
        """
        if not signals:
            return []
            
        optimized_signals = []
        
        # Calculate correlations between pairs
        correlations = self._calculate_correlations(signals, market_data)
        
        # Sort signals by confidence
        signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        for signal in signals:
            # Skip if confidence is too low
            if signal.confidence < self.confidence_threshold:
                continue
            
            # Check correlation with existing positions
            if not self._validate_correlation(signal.pair, current_positions, correlations):
                logger.info(f"Rejected {signal.pair} signal due to correlation")
                continue
            
            # Check maximum positions
            if len(current_positions) >= self.risk_profile.max_open_trades:
                logger.info("Maximum open positions reached")
                break
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal, account_balance, current_positions
            )
            
            if position_size > 0:
                # Create optimized signal
                optimized_signal = Signal(
                    pair=signal.pair,
                    timeframe=signal.timeframe,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    timestamp=signal.timestamp,
                    confidence=signal.confidence,
                    volume=position_size
                )
                optimized_signals.append(optimized_signal)
        
        return optimized_signals
    
    def _calculate_position_size(self,
                               signal: Signal,
                               account_balance: float,
                               current_positions: Dict[str, Dict]) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            current_positions: Currently open positions
            
        Returns:
            Position size in units
        """
        # Calculate base risk amount
        risk_per_trade = account_balance * (self.risk_profile.max_risk_per_trade / 100)
        
        # Adjust for confidence
        risk_amount = risk_per_trade * (signal.confidence / self.confidence_threshold)
        
        # Calculate stop loss distance
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance == 0:
            return 0
            
        # Calculate base position size
        position_size = risk_amount / stop_distance
        
        # Adjust for existing risk
        total_risk = sum(pos['risk_amount'] for pos in current_positions.values())
        available_risk = account_balance * (self.risk_profile.max_drawdown / 100) - total_risk
        
        if available_risk <= 0:
            return 0
            
        # Scale position size if needed
        position_risk = position_size * stop_distance
        if position_risk > available_risk:
            position_size *= available_risk / position_risk
            
        return position_size
    
    def _calculate_correlations(self,
                              signals: List[Signal],
                              market_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix for pairs.
        
        Args:
            signals: List of signals
            market_data: Market data dictionary
            
        Returns:
            Dictionary of pair correlations
        """
        correlations = {}
        pairs = list(set(signal.pair for signal in signals))
        
        for pair1 in pairs:
            correlations[pair1] = {}
            for pair2 in pairs:
                if pair1 == pair2:
                    correlations[pair1][pair2] = 1.0
                    continue
                
                try:
                    # Get price data
                    data1 = market_data[pair1]['1H']['close']
                    data2 = market_data[pair2]['1H']['close']
                    
                    # Calculate correlation on last 100 periods
                    corr = data1.tail(100).corr(data2.tail(100))
                    correlations[pair1][pair2] = corr
                    
                except Exception as e:
                    logger.warning(f"Error calculating correlation for {pair1}-{pair2}: {e}")
                    correlations[pair1][pair2] = 0.0
                    
        return correlations
    
    def _validate_correlation(self,
                            pair: str,
                            current_positions: Dict[str, Dict],
                            correlations: Dict[str, Dict[str, float]]) -> bool:
        """
        Check if pair correlation with existing positions is acceptable.
        
        Args:
            pair: Currency pair to validate
            current_positions: Currently open positions
            correlations: Correlation matrix
            
        Returns:
            True if correlation is acceptable, False otherwise
        """
        if not current_positions:
            return True
            
        for pos_pair in current_positions:
            if pos_pair == pair:
                return False
                
            try:
                correlation = abs(correlations[pair][pos_pair])
                if correlation > self.risk_profile.correlation_limit:
                    return False
            except KeyError:
                continue
                
        return True