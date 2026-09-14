import http.client
import json

conn = http.client.HTTPConnection('localhost', 8080)
headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}

conn.request('POST', '/mcp', json.dumps({
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'protocolVersion': '2024-11-05',
        'capabilities': {},
        'clientInfo': {'name': 'test', 'version': '1.0'}
    }
}), headers)
conn.getresponse().read()

conn.request('POST', '/mcp', json.dumps({
    'jsonrpc': '2.0',
    'id': 2,
    'method': 'tools/call',
    'params': {
        'name': 'execute_commands',
        'arguments': {
            'commands': ['execute if weather rain run tag @e[type=bee,tag=fly_brain_active,limit=1] add in_rain'],
            'validate_safety': False
        }
    }
}), headers)
resp = conn.getresponse().read().decode()
print("Response:", resp)
