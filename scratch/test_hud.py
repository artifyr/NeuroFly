import http.client, json

conn = http.client.HTTPConnection('localhost', 8080, timeout=3.0)
headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}

cmd1 = 'title @a actionbar [{"text":"Line 1: FAFB Bee HP: 20\\nLine 2: Spikes: 1,234 | Energy: 85%","color":"gold"}]'

cmd2_times = 'title @a times 0 10 5'
cmd2_title = 'title @a title [{"text":"[FAFB Bee] HP: 20","color":"gold","bold":true}]'
cmd2_sub = 'title @a subtitle [{"text":"[FORAGING] | Spikes: 1,234 | Energy: 92%","color":"aqua"}]'

for name, cmds in [('actionbar newline', [cmd1]), ('title + subtitle', [cmd2_times, cmd2_title, cmd2_sub])]:
    payload = json.dumps({
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': 'execute_commands', 'arguments': {'commands': cmds}}
    })
    conn.request('POST', '/mcp', payload, headers)
    resp = conn.getresponse()
    print(name, '->', resp.read().decode('utf-8')[:120])
