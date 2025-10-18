"""
Helper class for handling trade events and tracking trade metrics.
"""
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeEvent:
    timestamp: datetime
    trade_id: str
    pair: str
    price: float
    volume: float
    direction: str
    stop_loss: float
    take_profit: float
    event_type: str  # 'entry' or 'exit'
    pnl: float = 0.0
    status: str = 'open'

class TradeTracker:
    """Track and manage trade events and their metrics."""
    def __init__(self):
        self.open_trades = {}  # trade_id -> TradeEvent
        self.closed_trades = []  # list of completed trade records
        self._latest_equity = 0.0
    
    def record_entry(self, trade_id: str, pair: str, timestamp: datetime,
                    price: float, volume: float, direction: str,
                    stop_loss: float, take_profit: float) -> None:
        """Record a new trade entry."""
        event = TradeEvent(
            timestamp=timestamp,
            trade_id=trade_id,
            pair=pair,
            price=price,
            volume=volume,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            event_type='entry'
        )
        self.open_trades[trade_id] = event
    
    def record_exit(self, trade_id: str, timestamp: datetime,
                   price: float, pnl: float) -> None:
        """Record a trade exit."""
        if trade_id not in self.open_trades:
            return
            
        entry = self.open_trades[trade_id]
        trade_record = {
            'trade_id': trade_id,
            'pair': entry.pair,
            'timestamp': entry.timestamp,
            'exit_time': timestamp,
            'direction': entry.direction,
            'entry_price': entry.price,
            'exit_price': price,
            'volume': entry.volume,
            'stop_loss': entry.stop_loss,
            'take_profit': entry.take_profit,
            'pnl': pnl,
            'status': 'closed'
        }
        
        self.closed_trades.append(trade_record)
        del self.open_trades[trade_id]
        self._latest_equity += pnl
    
    def get_trades(self) -> list:
        """Get all completed trades."""
        return self.closed_trades
    
    def get_open_trades(self) -> list:
        """Get currently open trades."""
        return list(self.open_trades.values())
    
    def get_latest_equity(self) -> float:
        """Get current equity level."""
        return self._latest_equity