"""
Broker-specific configuration settings.
"""
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExnessBrokerConfig:
    """Exness broker-specific settings."""
    server_demo: str = "Exness-MT5Trial"  # Demo server
    server_real: str = "Exness-MT5Real"   # Real server
    common_pairs: List[str] = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", 
        "USDCAD", "NZDUSD", "USDCHF", "XAUUSD"
    ]
    point_multiplier: Dict[str, float] = {
        "EURUSD": 100000,
        "GBPUSD": 100000,
        "USDJPY": 1000,
        "AUDUSD": 100000,
        "USDCAD": 100000,
        "NZDUSD": 100000,
        "USDCHF": 100000,
        "XAUUSD": 10
    }
    min_lots: Dict[str, float] = {
        "EURUSD": 0.01,
        "GBPUSD": 0.01,
        "USDJPY": 0.01,
        "AUDUSD": 0.01,
        "USDCAD": 0.01,
        "NZDUSD": 0.01,
        "USDCHF": 0.01,
        "XAUUSD": 0.01
    }
    pip_value: Dict[str, float] = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "AUDUSD": 0.0001,
        "USDCAD": 0.0001,
        "NZDUSD": 0.0001,
        "USDCHF": 0.0001,
        "XAUUSD": 0.01
    }