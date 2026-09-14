import http.client
import json
import time

conn = http.client.HTTPConnection("localhost", 8080)
headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

def call(cmds):
    conn.request("POST", "/mcp", json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "execute_commands", "arguments": {"commands": cmds}},
    }), headers)
    return json.loads(conn.getresponse().read().decode("utf-8"))

# Clear old
call(["tp @e[tag=anim_test] ~ -200 ~"])

# 1. Spawn normal bee
res = call([
    'summon bee ~ ~2 ~ {CustomName:\'"NormalBee"\',CustomNameVisible:1b,Tags:["anim_test"]}'
])
print("Normal bee:", res)
