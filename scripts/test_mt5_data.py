import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

def test_mt5_data():
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed!")
        return

    try:
        # Connect to account (replace with your demo account details)
        connected = mt5.login(
            login=0,        # Replace with your account number
            password="",    # Replace with your password
            server="Exness-MT5Trial"
        )

        if not connected:
            print(f"Connection failed! Error: {mt5.last_error()}")
            return

        # Get account info to verify connection
        account_info = mt5.account_info()
        if account_info is not None:
            print(f"Connected to account #{account_info.login}")
            print(f"Balance: {account_info.balance}")
            print(f"Equity: {account_info.equity}\n")

        # Get real-time data for EURUSD
        symbol = "EURUSD"
        
        print(f"Getting real-time data for {symbol}")
        print("Press Ctrl+C to stop\n")

        while True:
            # Get the current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                print(f"Bid: {tick.bid:.5f}")
                print(f"Ask: {tick.ask:.5f}")
                print(f"Volume: {tick.volume}\n")
            
            # Get the last 5 minutes of M1 data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                print("Last 5 minutes of M1 data:")
                print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_string())
                print("\nWaiting 5 seconds...\n")
            
            time.sleep(5)  # Wait 5 seconds before next update

    except KeyboardInterrupt:
        print("\nStopping data collection...")

    finally:
        # Clean up
        mt5.shutdown()
        print("MT5 connection closed")

if __name__ == "__main__":
    print("Testing MT5 real-time data connection...")
    test_mt5_data()