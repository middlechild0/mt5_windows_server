# The AI Trading Project

This project implements an AI-powered trading system with MetaTrader 5 integration.

## Project Structure

The project is organized into 5 main folders:

### src/
Contains all source code files:
- mt5/ - MetaTrader 5 server implementation
- utils/ - Core utilities and components
  - backtest_engine.py - Backtesting functionality
  - broker_config.py - Broker configuration
  - config.py - General configuration
  - data_handler.py - Data management
  - live_trade_executor.py - Live trading execution
  - market_data_stream.py - Market data streaming
  - model_trainer.py - AI model training
  - mt5_client.py - MT5 client implementation
  - mt5_connector.py - MT5 connection management
  - performance_monitor.py - Performance tracking
  - prediction_pipeline.py - AI prediction pipeline
  - risk_manager.py - Risk management
  - signal_generator.py - Trading signal generation
  - signal_optimizer.py - Signal optimization
  - strategy_base.py - Base strategy class
  - trade_executor.py - Trade execution
  - trade_tracker.py - Trade tracking
  - training_pipeline.py - AI training pipeline

### data/
- raw/ - Raw market data
- processed/ - Processed data for analysis
- checkpoints/ - Model checkpoints
- logs/ - System and trading logs

### docs/
Documentation files including:
- Technical documentation
- Trading strategies
- API references
- Setup guides

### tests/
Unit tests and integration tests for all components

### scripts/
Utility scripts for tasks like:
- Database initialization
- Environment setup
- Data processing
- Model training

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables
3. Run tests: `python -m pytest tests/`
4. Start trading system: `python src/main.py`

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
- Trading parameters
- Safety limits
- Logging configuration

## Maintenance

- Regular data archival
- Log rotation
- Database backup
- Model versioning
- Performance monitoring