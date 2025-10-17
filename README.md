# Project Structure Overview

## Directory Structure

```
project_root/
├── data/
│   ├── raw/           # Original unprocessed data files
│   ├── processed/     # Cleaned and preprocessed data
│   ├── backtest/      # Data organized for backtesting
│   ├── live/          # Real-time trading data
│   └── archive/       # Historical data archive
│
├── knowledge/
│   ├── rules/         # Trading rules and strategy definitions
│   ├── models/        # Model architecture definitions
│   └── analytics/     # Analysis reports and insights
│
├── logs/
│   ├── trades/        # Trading activity logs
│   ├── system/        # System operation logs
│   └── errors/        # Error and warning logs
│
├── models/
│   ├── checkpoints/   # Model checkpoints during training
│   └── archive/       # Historical model versions
│
├── signals/
│   ├── backtest/      # Signals generated during backtesting
│   ├── live/          # Real-time trading signals
│   └── archive/       # Historical signals archive
│
├── src/
│   ├── core/          # Core system components
│   ├── utils/         # Utility functions and helpers
│   ├── strategies/    # Trading strategy implementations
│   └── config.py      # System configuration
│
└── sqlite_db/         # SQLite database files

## Code Organization

### Core Components (src/core/)
- Data processing pipeline
- Trading engine
- Risk management system
- Signal generation
- Model training and inference

### Utilities (src/utils/)
- File operations
- Data validation
- Type checking
- Logging utilities
- Database operations

### Strategies (src/strategies/)
- ICT trading strategies
- Custom indicator implementations
- Signal filters and validation

## Data Flow

1. Raw Data Input
   - Price data files → data/raw/
   - Cleaning and validation
   - Storage in processed format

2. Strategy Processing
   - Load processed data
   - Apply trading rules
   - Generate signals
   - Store in signals directory

3. Model Training
   - Feature extraction
   - Model training
   - Checkpoint saving
   - Version control

4. Live Trading
   - Real-time data processing
   - Signal generation
   - Risk management
   - Trade execution
   - Logging and monitoring

## Safety Features

- Automatic data backup
- Version control for models
- Error logging and monitoring
- Data validation at each step
- Risk management checks

## Configuration Management

All system parameters are managed in src/config.py:
- Database settings
- Directory paths
- Trading parameters
- Safety limits
- Logging configuration

## Maintenance

- Regular data archival
- Log rotation
- Database backup
- Model versioning
- Performance monitoring