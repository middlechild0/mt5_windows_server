# Project Configuration
PROJECT_CONFIG = {
    # Database Configuration
    "database": {
        "path": "../sqlite_db/trading_data.db",
        "backup_dir": "../backups/database"
    },
    
    # Data Directories
    "data_dirs": {
        "raw": "../data/raw",
        "processed": "../data/processed",
        "backtest": "../data/backtest",
        "live": "../data/live",
        "archive": "../data/archive"
    },
    
    # Models
    "model_dirs": {
        "base": "../models",
        "checkpoints": "../models/checkpoints",
        "archive": "../models/archive"
    },
    
    # Signals
    "signal_dirs": {
        "base": "../signals",
        "backtest": "../signals/backtest",
        "live": "../signals/live",
        "archive": "../signals/archive"
    },
    
    # Logging
    "log_dirs": {
        "base": "../logs",
        "trades": "../logs/trades",
        "system": "../logs/system",
        "errors": "../logs/errors",
        "backtest": "../logs/backtest"
    },
    
    # Knowledge Base
    "knowledge_dirs": {
        "base": "../knowledge",
        "rules": "../knowledge/rules",
        "models": "../knowledge/models",
        "analytics": "../knowledge/analytics"
    },
    
    # Pair-specific settings
    "pairs": {
        "XAUUSD": {
            "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
            "data_retention_days": 365,
            "min_history_required": 30
        }
    },
    
    # System Parameters
    "system": {
        "max_pairs": 5,
        "backup_interval_hours": 24,
        "cleanup_interval_days": 30,
        "max_concurrent_processes": 4
    },
    
    # Safety Parameters
    "safety": {
        "max_daily_trades": 10,
        "max_concurrent_trades": 3,
        "max_daily_drawdown_percent": 2,
        "max_position_size_percent": 1
    }
}