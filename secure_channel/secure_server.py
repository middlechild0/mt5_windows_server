"""
Secure WebSocket server for bidirectional communication between systems.
Enables remote code execution, system control, and trade management.
"""
import asyncio
import websockets
import json
import ssl
import jwt
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
import hashlib
import base64

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

logger = setup_logger('secure_channel', 'secure_channel.log')

# Security settings
SECRET_KEY = os.getenv('SECURE_CHANNEL_KEY', 'your-secret-key-here')  # Change in production
TOKEN_EXPIRY = timedelta(hours=24)

class SecureChannel:
    def __init__(self):
        self.clients = {}  # Store connected clients
        self.authorized_clients = set()  # Store authorized client IDs
        
    async def authenticate(self, websocket, path):
        """Authenticate incoming connections"""
        try:
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if 'token' not in auth_data:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Authentication required'
                }))
                return False
                
            # Verify JWT token
            try:
                payload = jwt.decode(auth_data['token'], SECRET_KEY, algorithms=['HS256'])
                client_id = payload['client_id']
                self.authorized_clients.add(client_id)
                self.clients[websocket] = client_id
                logger.info(f"Client {client_id} authenticated successfully")
                return True
            except jwt.InvalidTokenError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid token'
                }))
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False

    async def handle_command(self, websocket, command_data):
        """Handle incoming commands"""
        try:
            command_type = command_data.get('type')
            payload = command_data.get('payload', {})
            client_id = self.clients.get(websocket)

            response = {
                'type': 'response',
                'command': command_type,
                'status': 'success'
            }

            if command_type == 'code_update':
                # Handle code updates
                file_path = payload.get('file_path')
                content = payload.get('content')
                if file_path and content:
                    await self.handle_code_update(file_path, content)
                    response['message'] = f"Code updated: {file_path}"
                
            elif command_type == 'execute_trade':
                # Handle trade execution
                await self.handle_trade(payload)
                response['message'] = "Trade executed successfully"
                
            elif command_type == 'system_status':
                # Get system status
                status = await self.get_system_status()
                response['payload'] = status
                
            elif command_type == 'sync_files':
                # Handle file synchronization
                files = await self.sync_files(payload.get('paths', []))
                response['payload'] = {'files': files}

            logger.info(f"Command {command_type} executed successfully for client {client_id}")
            await websocket.send(json.dumps(response))

        except Exception as e:
            error_msg = f"Command execution error: {str(e)}"
            logger.error(error_msg)
            await websocket.send(json.dumps({
                'type': 'error',
                'command': command_type,
                'message': error_msg
            }))

    async def handle_code_update(self, file_path, content):
        """Safely handle code updates"""
        try:
            # Ensure the file path is within our project directory
            safe_path = Path(file_path).resolve()
            if not str(safe_path).startswith(str(Path.cwd())):
                raise ValueError("Invalid file path")

            # Create backup
            if safe_path.exists():
                backup_path = safe_path.with_suffix(f"{safe_path.suffix}.bak")
                safe_path.rename(backup_path)

            # Write new content
            safe_path.write_text(content)
            logger.info(f"Updated file: {file_path}")

        except Exception as e:
            logger.error(f"Code update error: {str(e)}")
            raise

    async def handle_trade(self, trade_data):
        """Handle trade execution"""
        # Implement trade execution logic here
        logger.info(f"Processing trade: {trade_data}")
        pass

    async def get_system_status(self):
        """Get current system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'connected_clients': len(self.clients),
            'authorized_clients': len(self.authorized_clients)
        }

    async def sync_files(self, paths):
        """Synchronize files between systems"""
        synced_files = []
        for path in paths:
            try:
                file_path = Path(path).resolve()
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text()
                    hash_value = hashlib.sha256(content.encode()).hexdigest()
                    synced_files.append({
                        'path': str(path),
                        'hash': hash_value,
                        'last_modified': file_path.stat().st_mtime
                    })
            except Exception as e:
                logger.error(f"File sync error for {path}: {str(e)}")
        return synced_files

    async def handler(self, websocket, path):
        """Main WebSocket connection handler"""
        try:
            if not await self.authenticate(websocket, path):
                return

            async for message in websocket:
                try:
                    command_data = json.loads(message)
                    await self.handle_command(websocket, command_data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Message handling error: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {self.clients.get(websocket)}")
        finally:
            if websocket in self.clients:
                client_id = self.clients[websocket]
                self.authorized_clients.discard(client_id)
                del self.clients[websocket]

async def start_server():
    """Start the secure WebSocket server"""
    channel = SecureChannel()
    
    # Setup SSL context for secure WebSocket (wss://)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')  # You'll need to generate these
    
    server = await websockets.serve(
        channel.handler,
        '0.0.0.0',  # Listen on all interfaces
        8765,       # WebSocket port
        ssl=ssl_context
    )
    
    logger.info("Secure channel server started")
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(start_server())