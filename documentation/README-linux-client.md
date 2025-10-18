# MT5 Linux Trading Client

This is the Linux client component of the distributed MT5 trading system. It works in conjunction with the MT5 Windows server to enable automated trading from Linux machines.

## Architecture

The system consists of two main components:

1. **Windows MT5 Server** (in separate repository)
   - Runs on Windows machine with MT5 installed
   - Provides REST API for trading operations
   - Handles direct communication with MT5

2. **Linux Trading Client** (this repository)
   - Runs on Linux machine
   - Communicates with Windows MT5 server via HTTP
   - Makes trading decisions
   - Sends trading commands to server

## Setup

1. First ensure the Windows MT5 server is running and accessible.

2. Install dependencies:
```bash
pip install -r requirements-linux-client.txt
```

3. Configure the client:
   - Copy `.env.example` to `.env`
   - Set your Windows server IP and other parameters:
```env
MT5_SERVER_HOST=192.168.1.xxx  # Replace with your Windows IP
MT5_SERVER_PORT=5000
MT5_API_KEY=your_api_key  # If configured on server
```

## Usage

### Basic Usage

```python
from src.core.mt5_client import MT5Client
from src.core.config import config

# Initialize client
client = MT5Client(config.SERVER_HOST, config.SERVER_PORT)

# Get account info
account_info = client.get_account_info()
print(f"Balance: {account_info['balance']}")

# Get current price
price = client.get_price("EURUSD")
print(f"EURUSD Bid: {price['bid']}, Ask: {price['ask']}")

# Get open positions
positions = client.get_positions()
print(f"Open positions: {len(positions)}")
```

### Running Example Trader

An example trading script is provided in `examples/mt5_example_trader.py`. To run it:

```bash
python examples/mt5_example_trader.py
```

The example trader:
- Monitors prices for configured symbols
- Tracks open positions
- Logs account status
- Can be extended with custom trading logic

## Directory Structure

```
src/
├── core/
│   ├── mt5_client.py     # Main MT5 client class
│   ├── config.py         # Configuration settings
examples/
├── mt5_example_trader.py # Example trading script
```

## Client API Reference

### MT5Client Methods

- `get_account_info()`: Get account balance, equity, etc.
- `get_price(symbol)`: Get current bid/ask for symbol
- `get_positions()`: List all open positions
- More methods available in mt5_client.py

## Error Handling

The client includes error handling for:
- Connection issues
- API errors
- Invalid responses

All methods return Optional types, returning None on failure with logged errors.

## Security Notes

1. Use firewall rules to restrict access to the MT5 server
2. Consider using API key authentication
3. Use secure network for client-server communication
4. Never expose MT5 server to public internet