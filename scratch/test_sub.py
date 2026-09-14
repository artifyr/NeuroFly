import http.client, json

conn = http.client.HTTPConnection('localhost', 8080, timeout=3.0)
headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}

cmds = [
    'title @a times 0 40 10',
    'title @a title {"text":""}',
    'title @a subtitle [{"text":"[FAFB Bee] HP: 20 [♥♥♥♥♥]  [FORAGING]\\n","color":"gold","bold":true},{"text":"Spikes: 1,420 | Energy: 85% | PAM-DA: 0.90 | Cover: Open","color":"aqua"}]'
]

payload = json.dumps({
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'tools/call',
    'params': {'name': 'execute_commands', 'arguments': {'commands': cmds}}
})
conn.request('POST', '/mcp', payload, headers)
resp = conn.getresponse()
print(resp.read().decode('utf-8'))
