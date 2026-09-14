import http.client, json, time

conn = http.client.HTTPConnection('localhost', 8080, timeout=3.0)
headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}

tp_cmd = 'tp @e[type=bee,limit=1] ~ ~ ~'
hud_cmd = 'title @a actionbar [{"text":"Test"}]'

for name, cmds in [('tp only', [tp_cmd]), ('tp + title', [tp_cmd, hud_cmd]), ('title only', [hud_cmd])]:
    for i in range(3):
        t0 = time.perf_counter()
        payload = json.dumps({
            'jsonrpc': '2.0',
            'id': 100 + i,
            'method': 'tools/call',
            'params': {
                'name': 'execute_commands',
                'arguments': {'commands': cmds}
            }
        })
        conn.request('POST', '/mcp', payload, headers)
        resp = conn.getresponse()
        data = json.loads(resp.read().decode('utf-8'))
        t1 = time.perf_counter()
        print(f'{name} {i}: {(t1-t0)*1000:.1f}ms')
