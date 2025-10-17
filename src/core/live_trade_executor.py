"""
Live trading execution and order management.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, UTC
import logging
from dataclasses import dataclass
from enum import Enum
import uuid

from src.core.strategy_base import Signal
from src.core.market_data_stream import MarketUpdate

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"

class OrderStatus(Enum):
    """Possible order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Trading order details."""
    id: str
    pair: str
    order_type: OrderType
    direction: str
    volume: float
    price: float
    status: OrderStatus
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    metadata: Optional[Dict] = None

@dataclass
class Position:
    """Open trading position details."""
    id: str
    pair: str
    direction: str
    volume: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    metadata: Optional[Dict] = None

class LiveTradeExecutor:
    def __init__(self, account_config: Dict):
        """
        Initialize trade executor.
        
        Args:
            account_config: Trading account configuration
        """
        self.account_config = account_config
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
        # Performance tracking
        self.initial_balance = account_config.get('initial_balance', 0.0)
        self.current_balance = self.initial_balance
        self.equity_curve: List[Dict] = []
        
    def place_market_order(self,
                          signal: Signal,
                          current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Place a market order based on a trading signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            (success, order_id)
        """
        # Validate signal
        if not self._validate_order(signal):
            return False, None
            
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            pair=signal.pair,
            order_type=OrderType.MARKET,
            direction=signal.direction,
            volume=signal.volume,
            price=current_price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(UTC),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata={'signal_confidence': signal.confidence}
        )
        
        # Process order
        success = self._process_market_order(order)
        if success:
            return True, order_id
        return False, None
    
    def update_positions(self, market_update: MarketUpdate) -> List[str]:
        """
        Update positions with new market data.
        
        Args:
            market_update: Latest market data
            
        Returns:
            List of closed position IDs
        """
        closed_positions = []
        
        # Get mid price
        mid_price = (market_update.bid + market_update.ask) / 2
        
        # Update each position
        for pos_id, position in list(self.positions.items()):
            if position.pair != market_update.pair:
                continue
                
            # Update unrealized P&L
            price_diff = round((mid_price - position.entry_price) * 10000) / 10000  # Round to 4 decimal places
            if position.direction == 'sell':
                price_diff = -price_diff
            position.unrealized_pnl = round(price_diff * position.volume, 2)  # Round PnL to 2 decimal places
            
            # Check stop loss
            if (position.direction == 'buy' and mid_price <= position.stop_loss) or \
               (position.direction == 'sell' and mid_price >= position.stop_loss):
                self._close_position(pos_id, mid_price, market_update.timestamp, 'stop_loss')
                closed_positions.append(pos_id)
                continue
                
            # Check take profit
            if (position.direction == 'buy' and mid_price >= position.take_profit) or \
               (position.direction == 'sell' and mid_price <= position.take_profit):
                self._close_position(pos_id, mid_price, market_update.timestamp, 'take_profit')
                closed_positions.append(pos_id)
                
        return closed_positions
    
    def get_position(self, pair: str) -> Optional[Position]:
        """Get current position for a pair."""
        for position in self.positions.values():
            if position.pair == pair:
                return position
        return None
    
    def _validate_order(self, signal: Signal) -> bool:
        """Validate order parameters."""
        if not signal.volume or signal.volume <= 0:
            logger.warning("Invalid order volume")
            return False
            
        if signal.pair in [p.pair for p in self.positions.values()]:
            logger.warning(f"Position already exists for {signal.pair}")
            return False
            
        return True
    
    def _process_market_order(self, order: Order) -> bool:
        """Process a market order."""
        try:
            # Simulate immediate fill
            order.status = OrderStatus.FILLED
            order.fill_price = order.price
            order.fill_time = datetime.now(UTC)
            
            # Create position
            position = Position(
                id=str(uuid.uuid4()),
                pair=order.pair,
                direction=order.direction,
                volume=order.volume,
                entry_price=order.fill_price,
                entry_time=order.fill_time,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                metadata=order.metadata
            )
            
            # Store order and position
            self.orders[order.id] = order
            self.positions[position.id] = position
            self.order_history.append(order)
            
            # Update equity
            self._update_equity(order.timestamp)
            
            logger.info(f"Opened position for {order.pair}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing market order: {e}")
            return False
    
    def _close_position(self,
                       position_id: str,
                       exit_price: float,
                       timestamp: datetime,
                       reason: str) -> None:
        """Close a position and record the result."""
        position = self.positions[position_id]
        
        # Calculate P&L
        price_diff = exit_price - position.entry_price
        if position.direction == 'sell':
            price_diff = -price_diff
        pnl = price_diff * position.volume
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade result
        trade_result = {
            'pair': position.pair,
            'direction': position.direction,
            'volume': position.volume,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'pnl': pnl,
            'reason': reason
        }
        
        # Update equity curve
        self._update_equity(timestamp, trade_result)
        
        # Remove position
        del self.positions[position_id]
        logger.info(f"Closed position for {position.pair}, PnL: {pnl:.2f}")
    
    def _update_equity(self,
                      timestamp: datetime,
                      trade_result: Optional[Dict] = None) -> None:
        """Update equity curve with new balance point."""
        equity_point = {
            'timestamp': timestamp,
            'balance': self.current_balance,
            'equity': self.current_balance + sum(p.unrealized_pnl for p in self.positions.values())
        }
        
        if trade_result:
            equity_point['trade'] = trade_result
            
        self.equity_curve.append(equity_point)