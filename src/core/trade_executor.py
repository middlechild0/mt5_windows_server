"""
Trading executor that handles order management and execution.
"""
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from src.core.strategy_base import Signal
from src.core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class Order:
    id: str
    timestamp: datetime
    pair: str
    type: str  # 'market', 'limit', 'stop'
    side: str  # 'buy', 'sell'
    price: float
    volume: float
    stop_loss: float
    take_profit: float
    status: str  # 'pending', 'filled', 'cancelled'
    signal_id: str

@dataclass
class Position:
    order_id: str
    pair: str
    side: str
    entry_price: float
    current_price: float
    volume: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    entry_time: datetime
    signal_id: str

class TradeExecutor:
    def __init__(self, risk_manager: RiskManager, trade_tracker=None):
        self.risk_manager = risk_manager
        self.trade_tracker = trade_tracker
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
    def place_order(self, signal: Signal, volume: float) -> Tuple[bool, str]:
        """
        Place a new order based on signal.
        
        Returns:
            (success, order_id/error_message)
        """
        # Generate unique order ID
        order_id = f"{signal.pair}_{signal.timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # Create order object
        order = Order(
            id=order_id,
            timestamp=signal.timestamp,
            pair=signal.pair,
            type='market',  # For now, only market orders
            side=signal.direction,
            price=signal.entry_price,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status='pending',
            signal_id=signal.signal_type
        )
        
        # Store order
        self.active_orders[order_id] = order
        
        logger.info(f"Placed {order.side} order for {order.pair}: "
                   f"Price={order.price:.5f}, Volume={order.volume:.3f}")
        
        return True, order_id
    
    def update_order(self, order_id: str, new_status: str,
                    fill_price: Optional[float] = None) -> None:
        """Update order status and handle order fills."""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found")
            return
            
        order = self.active_orders[order_id]
        order.status = new_status
        
        if new_status == 'filled':
            # Create position from filled order
            position = Position(
                order_id=order_id,
                pair=order.pair,
                side=order.side,
                entry_price=fill_price or order.price,
                current_price=fill_price or order.price,
                volume=order.volume,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                unrealized_pnl=0.0,
                entry_time=datetime.now(),
                signal_id=order.signal_id
            )
            
            self.positions[order.pair] = position
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            logger.info(f"Order {order_id} filled at {position.entry_price:.5f}")
    
    def update_positions(self, current_prices: Dict[str, float],
                        current_time: datetime) -> List[str]:
        """
        Update all positions with current prices and check for stops.
        
        Returns:
            List of pairs where position was closed
        """
        closed_positions = []
        
        for pair, position in list(self.positions.items()):
            if pair not in current_prices:
                continue
                
            current_price = current_prices[pair]
            position.current_price = current_price
            
            # Calculate unrealized P&L
            price_diff = current_price - position.entry_price
            if position.side == 'sell':
                price_diff = -price_diff
            position.unrealized_pnl = price_diff * position.volume
            
            # Check stop loss and take profit
            if position.side == 'buy':
                if current_price <= position.stop_loss or current_price >= position.take_profit:
                    self._close_position(pair, current_price, current_time)
                    closed_positions.append(pair)
            else:  # sell
                if current_price >= position.stop_loss or current_price <= position.take_profit:
                    self._close_position(pair, current_price, current_time)
                    closed_positions.append(pair)
        
        return closed_positions
    
    def _close_position(self, pair: str, price: float, timestamp: datetime) -> None:
        """Close a position and log the result."""
        position = self.positions[pair]
        
        # Calculate realized P&L
        price_diff = price - position.entry_price
        if position.side == 'sell':
            price_diff = -price_diff
        pnl = price_diff * position.volume
        
        logger.info(f"Closed position for {pair}: Entry={position.entry_price:.5f}, "
                   f"Exit={price:.5f}, PnL={pnl:.2f}")
        
        # Update risk manager
        self.risk_manager.update_metrics(pnl, timestamp, pnl)
        
        # Record trade closure
        self.trade_tracker.record_exit(position.order_id, timestamp, price, pnl)
        
        # Remove position
        del self.positions[pair]