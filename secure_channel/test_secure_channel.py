"""
Test script for secure channel communication
"""
import asyncio
import os
from secure_client import SecureChannelClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_secure_channel():
    """Test secure channel functionality"""
    client = SecureChannelClient(
        server_url='wss://localhost:8765',
        client_id='test-client',
        secret_key='your-secret-key-here'
    )
    
    # Test connection
    logger.info("Testing connection...")
    if await client.connect():
        logger.info("Connection successful!")
        
        # Test system status
        logger.info("Getting system status...")
        status = await client.get_system_status()
        logger.info(f"System status: {status}")
        
        # Test file sync
        logger.info("Testing file sync...")
        sync_result = await client.sync_files(['secure_server.py', 'secure_client.py'])
        logger.info(f"File sync result: {sync_result}")
        
        # Test code update
        test_code = 'print("Hello from secure channel!")'
        logger.info("Testing code update...")
        await client.update_code('test_output.py', test_code)
        logger.info("Code update complete")
        
        # Keep connection alive for a few seconds
        await asyncio.sleep(5)
        logger.info("Test complete!")

if __name__ == '__main__':
    asyncio.run(test_secure_channel())