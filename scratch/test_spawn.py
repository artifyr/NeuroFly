import http.client
import json

conn = http.client.HTTPConnection('localhost', 8080, timeout=2.0)
cmds = [
    'tp @e[tag=fly_brain_active] ~ -300 ~',
    'summon bee -37.0 -57.5 177.0 {CustomName:\'"FAFB Connectome Fly (138k SNN)"\',CustomNameVisible:1b,NoGravity:1b,Invulnerable:1b,PersistenceRequired:1b,Tags:["fly_brain_active"]}',
    'title @a actionbar [{"text":"[FlyBrain] ","color":"gold","bold":true},{"text":"Bee Spawned!","color":"green"}]'
]
payload = json.dumps({
    'jsonrpc': '2.0',
    'id': 3,
    'method': 'tools/call',
    'params': {'name': 'execute_commands', 'arguments': {'commands': cmds}}
})
conn.request('POST', '/mcp', payload, {'Content-Type': 'application/json', 'Accept': 'application/json'})
resp = conn.getresponse()
data = resp.read().decode('utf-8')
print('Execute result:', data)
conn.close()
