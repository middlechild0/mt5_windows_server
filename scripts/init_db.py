"""
Database initialization script.
Creates the SQLite database and all required tables.
"""
import sqlite3
from pathlib import Path
import sys

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.config import PROJECT_CONFIG

def init_database():
    """Initialize the database with all required tables."""
    db_path = Path(PROJECT_CONFIG['database']['path'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database (will create if doesn't exist)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create pairs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            pair TEXT PRIMARY KEY,
            last_update TIMESTAMP,
            data_path TEXT,
            timeframe TEXT,
            first_timestamp TIMESTAMP,
            last_timestamp TIMESTAMP,
            row_count INTEGER,
            status TEXT CHECK(status IN ('active', 'archived', 'invalid'))
        )
        """)
        
        # Create data_validation table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            timestamp TIMESTAMP,
            validation_type TEXT,
            status TEXT,
            details TEXT,
            FOREIGN KEY(pair) REFERENCES pairs(pair)
        )
        """)
        
        # Create trading_signals table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trading_signals (
            id TEXT PRIMARY KEY,
            pair TEXT,
            timestamp TIMESTAMP,
            signal_type TEXT,
            direction TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT,
            FOREIGN KEY(pair) REFERENCES pairs(pair)
        )
        """)
        
        # Create trades table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            pair TEXT,
            timestamp_utc TIMESTAMP,
            side TEXT CHECK(side IN ('buy', 'sell')),
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            lots REAL,
            result REAL,
            exit_price REAL,
            exit_timestamp TIMESTAMP,
            pnl REAL,
            status TEXT,
            raw_log TEXT,
            FOREIGN KEY(pair) REFERENCES pairs(pair)
        )
        """)
        
        # Create models table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            pair TEXT,
            model_path TEXT,
            last_trained TIMESTAMP,
            accuracy REAL,
            params TEXT,
            version TEXT,
            status TEXT,
            PRIMARY KEY (pair, version),
            FOREIGN KEY(pair) REFERENCES pairs(pair)
        )
        """)
        
        # Create meta table for system settings
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP
        )
        """)
        
        conn.commit()
        print("Database initialized successfully!")

def verify_database():
    """Verify all tables exist and have correct schema."""
    db_path = Path(PROJECT_CONFIG['database']['path'])
    if not db_path.exists():
        print("Database does not exist!")
        return False
        
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {
            'pairs', 'data_validation', 'trading_signals', 
            'trades', 'models', 'meta'
        }
        
        missing_tables = required_tables - tables
        if missing_tables:
            print(f"Missing tables: {missing_tables}")
            return False
            
        print("Database verification successful!")
        return True

if __name__ == "__main__":
    init_database()
    verify_database()