"""
Secure channel client for bidirectional communication.
Can be used on both Windows and Linux systems.
"""
import asyncio
import websockets
import json
import jwt
import ssl
import logging
from datetime import datetime
from pathlib import Path
import hashlib
from logging.handlers import RotatingFileHandler

# Setup logging
log_dir = Path("logs/secure_channel")
log_dir.mkdir(parents=True, exist_ok=True)

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=1024*1024,
        backupCount=5
    )
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger('secure_channel_client', 'client.log')

class SecureChannelClient:
    def __init__(self, server_url, client_id, secret_key):
        self.server_url = server_url
        self.client_id = client_id
        self.secret_key = secret_key
        self.websocket = None
        
    def generate_token(self):
        """Generate JWT token for authentication"""
        return jwt.encode(
            {
                'client_id': self.client_id,
                'timestamp': datetime.now().isoformat()
            },
            self.secret_key,
            algorithm='HS256'
        )

    async def connect(self):
        """Connect to secure channel server"""
        try:
            # Create SSL context that doesn't verify cert for local testing
            ssl_context = ssl.SSLContext()
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                self.server_url,
                ssl=ssl_context
            )
            
            # Authenticate
            auth_message = {
                'type': 'auth',
                'token': self.generate_token()
            }
            await self.websocket.send(json.dumps(auth_message))
            
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get('type') == 'error':
                raise Exception(f"Authentication failed: {response_data.get('message')}")
                
            logger.info("Connected and authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False

    async def update_code(self, file_path, content):
        """Update code on the remote system"""
        command = {
            'type': 'code_update',
            'payload': {
                'file_path': str(file_path),
                'content': content
            }
        }
        await self.send_command(command)

    async def execute_trade(self, trade_data):
        """Execute a trade on the remote system"""
        command = {
            'type': 'execute_trade',
            'payload': trade_data
        }
        await self.send_command(command)

    async def get_system_status(self):
        """Get current system status"""
        command = {
            'type': 'system_status'
        }
        return await self.send_command(command)

    async def sync_files(self, paths):
        """Synchronize files with the remote system"""
        command = {
            'type': 'sync_files',
            'payload': {
                'paths': [str(p) for p in paths]
            }
        }
        return await self.send_command(command)

    async def send_command(self, command):
        """Send command to server and wait for response"""
        try:
            await self.websocket.send(json.dumps(command))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Command error: {str(e)}")
            raise

    async def listen_for_updates(self):
        """Listen for updates from the server"""
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                logger.info(f"Received update: {data}")
                # Handle different types of updates
                if data.get('type') == 'system_update':
                    logger.info(f"System update: {data.get('payload')}")
                elif data.get('type') == 'trade_update':
                    logger.info(f"Trade update: {data.get('payload')}")
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed")
        except Exception as e:
            logger.error(f"Listen error: {str(e)}")

async def main():
    """Example usage of the secure channel client"""
    client = SecureChannelClient(
        server_url='wss://localhost:8765',
        client_id='test-client',
        secret_key='your-secret-key-here'  # Change in production
    )
    
    if await client.connect():
        # Start listening for updates in the background
        asyncio.create_task(client.listen_for_updates())
        
        # Example: Get system status
        status = await client.get_system_status()
        logger.info(f"System status: {status}")
        
        # Example: Update code
        await client.update_code(
            'test.py',
            'print("Hello, World!")'
        )
        
        # Example: Execute trade
        await client.execute_trade({
            'symbol': 'XAUUSD',
            'type': 'BUY',
            'volume': 0.01
        })
        
        # Keep the connection alive
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    asyncio.run(main())