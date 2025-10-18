"""
MT5 server providing REST API access to MetaTrader 5 functionality.
"""
import MetaTrader5 as mt5
from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import logging
import os
from logging.handlers import RotatingFileHandler
import config

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('logs/trades', exist_ok=True)
os.makedirs('logs/errors', exist_ok=True)
os.makedirs('logs/system', exist_ok=True)

# Setup different loggers for different purposes
def setup_logger(name, log_file, level=logging.INFO):
    handler = RotatingFileHandler(
        log_file,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# Initialize different loggers
system_logger = setup_logger('system', 'logs/system/mt5_system.log')
trade_logger = setup_logger('trades', 'logs/trades/mt5_trades.log')
error_logger = setup_logger('errors', 'logs/errors/mt5_errors.log', logging.ERROR)

app = Flask(__name__)

def connect_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        error_msg = f"MT5 initialization failed: {mt5.last_error()}"
        error_logger.error(error_msg)
        system_logger.error(error_msg)
        return False
    
    # Connect to account
    if not mt5.login(config.MT5_ACCOUNT, config.MT5_PASSWORD, config.MT5_SERVER):
        error_msg = f"MT5 login failed: {mt5.last_error()}"
        error_logger.error(error_msg)
        system_logger.error(error_msg)
        return False
        
    system_logger.info("MT5 connected successfully")
    return True

def check_api_key():
    """Verify API key if configured"""
    if config.API_KEY:
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != config.API_KEY:
            return False
    return True

@app.before_request
def before_request():
    """Check API key before each request"""
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

@app.route('/health')
def health_check():
    """Check if server is running and MT5 is connected"""
    if not mt5.initialize():
        return jsonify({
            "status": "error",
            "message": "MT5 not initialized"
        }), 500
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/account_info')
def get_account_info():
    """Get account information"""
    if not mt5.initialize():
        return jsonify({"error": "MT5 not initialized"}), 500
        
    info = mt5.account_info()
    if info is None:
        return jsonify({"error": "Could not get account info"}), 500
        
    return jsonify({
        "login": info.login,
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "leverage": info.leverage,
        "currency": info.currency
    })

@app.route('/price/<symbol>')
def get_price(symbol):
    """Get current price for a symbol"""
    system_logger.info(f"Price request received for symbol: {symbol}")
    
    if not mt5.initialize():
        error_msg = "MT5 not initialized when requesting price"
        error_logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
        
    # Select the symbol in Market Watch
    if not mt5.symbol_select(symbol, True):
        error_msg = f"Symbol {symbol} not found or not selected in Market Watch"
        error_logger.error(error_msg)
        return jsonify({"error": error_msg}), 404
        
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        error_msg = f"Could not get price for {symbol}"
        error_logger.error(error_msg)
        return jsonify({"error": error_msg}), 500
        
    system_logger.info(f"Price retrieved for {symbol}: Bid={tick.bid}, Ask={tick.ask}")
        
    return jsonify({
        "symbol": symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "last": tick.last,
        "volume": tick.volume,
        "time": datetime.fromtimestamp(tick.time).isoformat()
    })

@app.route('/positions')
def get_positions():
    """Get open positions"""
    if not mt5.initialize():
        return jsonify({"error": "MT5 not initialized"}), 500
        
    positions = mt5.positions_get()
    if positions is None:
        return jsonify({"error": "Could not get positions"}), 500
        
    return jsonify([{
        "ticket": pos.ticket,
        "symbol": pos.symbol,
        "type": "buy" if pos.type == 0 else "sell",
        "volume": pos.volume,
        "open_price": pos.price_open,
        "current_price": pos.price_current,
        "sl": pos.sl,
        "tp": pos.tp,
        "profit": pos.profit,
        "comment": pos.comment
    } for pos in positions])

@app.route('/place_order', methods=['POST'])
def place_order():
    """Place a new market order"""
    if not mt5.initialize():
        return jsonify({"error": "MT5 not initialized"}), 500
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    # Required parameters
    symbol = data.get('symbol')
    order_type = data.get('type')  # "buy" or "sell"
    volume = data.get('volume')
    
    # Optional parameters
    sl = data.get('sl', 0.0)
    tp = data.get('tp', 0.0)
    
    if not all([symbol, order_type, volume]):
        return jsonify({"error": "Missing required parameters"}), 400
        
    # Prepare the order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if order_type.lower() == "buy" else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(symbol).ask if order_type.lower() == "buy" else mt5.symbol_info_tick(symbol).bid,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 234000,
        "comment": "REST API Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "error": "Order failed",
            "retcode": result.retcode,
            "comment": result.comment
        }), 500
        
    return jsonify({
        "success": True,
        "order": result.order,
        "volume": result.volume,
        "price": result.price,
        "bid": result.bid,
        "ask": result.ask,
        "comment": result.comment
    })

@app.route('/close_position/<int:ticket>', methods=['POST'])
def close_position(ticket):
    """Close a specific position by ticket number"""
    if not mt5.initialize():
        return jsonify({"error": "MT5 not initialized"}), 500
        
    # Get position details
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return jsonify({"error": f"Position {ticket} not found"}), 404
        
    position = position[0]
    
    # Prepare close request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "REST API Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send close request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "error": "Close failed",
            "retcode": result.retcode,
            "comment": result.comment
        }), 500
        
    return jsonify({
        "success": True,
        "order": result.order,
        "volume": result.volume,
        "price": result.price
    })

if __name__ == '__main__':
    system_logger.info(f"Starting MT5 server on {config.HOST}:{config.PORT}")
    system_logger.info(f"Debug mode: {config.DEBUG}")
    
    if not connect_mt5():
        error_logger.error("Failed to start MT5 server")
        exit(1)
        
    system_logger.info("MT5 server started successfully")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)