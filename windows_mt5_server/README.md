# MT5 Trading Server
A simple server to expose MetaTrader 5 functionality via REST API.

## Requirements
- Windows OS
- Python 3.8 or higher
- MetaTrader 5 terminal installed
- MetaTrader 5 account (demo or live)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/mt5-trading-server.git
cd mt5-trading-server
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Edit `config.py` with your MT5 account details:
```python
MT5_ACCOUNT = your_account_number
MT5_PASSWORD = "your_password"
MT5_SERVER = "your_server"  # e.g., "Exness-MT5Trial"
```

4. Run the server:
```bash
python mt5_server.py
```

## API Endpoints

### GET /account_info
Returns account balance, equity, margin, etc.

### GET /price/<symbol>
Returns current bid/ask prices for the specified symbol.

### GET /positions
Returns all open positions.

## Security Notice
- Server runs on localhost by default
- Add authentication if exposing to network
- Keep your MT5 credentials secure

## Testing
Run tests with:
```bash
python -m pytest tests/
```

## Logging
Logs are written to `mt5_server.log`