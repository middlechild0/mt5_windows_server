import MetaTrader5 as mt5
from datetime import datetime

def test_exness_connection():
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed! Error: {mt5.last_error()}")
        return False

    try:
        # Connect to your Exness demo account
        login_result = mt5.login(
            login=0,  # We'll need your account number
            password="kRPzA43a",
            server="Exness-MT5Trial"
        )

        if not login_result:
            print(f"Login failed! Error: {mt5.last_error()}")
            return False

        # Get and display account info
        account_info = mt5.account_info()
        if account_info is not None:
            print("\n=== Account Information ===")
            print(f"Login: {account_info.login}")
            print(f"Name: {account_info.name}")
            print(f"Server: {account_info.server}")
            print(f"Balance: ${account_info.balance:.2f}")
            print(f"Equity: ${account_info.equity:.2f}")
            print(f"Margin: ${account_info.margin:.2f}")
            print(f"Free Margin: ${account_info.margin_free:.2f}")
            print(f"Leverage: 1:{account_info.leverage}")
            print("========================\n")

            # Get some symbol information (e.g., EURUSD)
            symbol = "EURUSD"
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                print(f"=== {symbol} Information ===")
                print(f"Bid: {symbol_info.bid:.5f}")
                print(f"Ask: {symbol_info.ask:.5f}")
                print(f"Spread: {symbol_info.spread} points")
                print(f"Trade Mode: {symbol_info.trade_mode}")
                print("========================\n")
            else:
                print(f"Failed to get {symbol} information")

            return True
        else:
            print("Failed to get account information")
            return False

    except Exception as e:
        print(f"Error occurred: {e}")
        return False

    finally:
        # Shut down connection to MT5
        mt5.shutdown()

if __name__ == "__main__":
    print("Testing connection to Exness demo account...")
    success = test_exness_connection()
    if success:
        print("Connection test completed successfully!")
    else:
        print("Connection test failed!")