"""
Configuration settings for MT5 server.
Load from environment variables or set directly.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MT5 Settings
MT5_ACCOUNT = int(os.getenv('MT5_ACCOUNT', '211678367'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'kRPzA43a')
MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5Trial')

# Server Settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Security
API_KEY = os.getenv('API_KEY', None)  # Optional API key for authentication