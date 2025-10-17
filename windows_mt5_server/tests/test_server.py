"""
Test suite for MT5 server functionality.
"""
import pytest
from mt5_server import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test server health check endpoint"""
    rv = client.get('/health')
    json_data = rv.get_json()
    assert rv.status_code == 200
    assert json_data['status'] == 'ok'
    assert 'timestamp' in json_data

def test_account_info(client):
    """Test account information endpoint"""
    rv = client.get('/account_info')
    json_data = rv.get_json()
    assert rv.status_code == 200
    assert 'balance' in json_data
    assert 'equity' in json_data
    assert 'leverage' in json_data

def test_price_endpoint(client):
    """Test price data endpoint"""
    rv = client.get('/price/EURUSD')
    json_data = rv.get_json()
    assert rv.status_code == 200
    assert 'bid' in json_data
    assert 'ask' in json_data
    assert 'time' in json_data

def test_positions(client):
    """Test positions endpoint"""
    rv = client.get('/positions')
    json_data = rv.get_json()
    assert rv.status_code == 200
    assert isinstance(json_data, list)

def test_place_order(client):
    """Test order placement endpoint"""
    order_data = {
        "symbol": "EURUSD",
        "type": "buy",
        "volume": 0.01
    }
    rv = client.post('/place_order',
                    data=json.dumps(order_data),
                    content_type='application/json')
    json_data = rv.get_json()
    assert rv.status_code == 200
    assert 'success' in json_data
    assert json_data['success'] == True