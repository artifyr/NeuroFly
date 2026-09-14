"""
Continuous Live Flight Controller for the FAFB Brain Bee in Minecraft.

Runs continuously:
- Communicates with FlyBrainSNN server via WebSocket (ws://localhost:8765).
- Continuously calculates realistic biological steering and forward wing propulsion.
- Animates flight trajectory, smooth yaw banking, and hover altitude.
- Outputs Minecraft command list for immediate execution or batch streaming.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


async def run_live_flight_session(steps: int = 30, delay: float = 0.1):
    """
    Connects to the SNN server and yields continuous flight waypoints.
    """
    x, y, z = -51.0, -58.5, 58.0
    yaw = 45.0
    speed = 0.28

    print(f"Connecting to FlyBrainSNN at ws://localhost:8765...")
    waypoints = []

    async with connect("ws://localhost:8765") as ws:
        print("Connected to SNN WebSocket.")
        for step in range(steps):
            # Dynamic sensory stimulus simulating visual optic motion
            left_eye = 0.5 + 0.4 * math.sin(step * 0.35)
            right_eye = 0.5 - 0.4 * math.sin(step * 0.35)

            req = {"sensors": [left_eye, right_eye]}
            await ws.send(json.dumps(req))

            raw_resp = await ws.recv()
            telemetry = json.loads(raw_resp)

            turn = float(telemetry.get("turn", 0.0))
            forward = float(telemetry.get("forward", 0.0))
            spikes = int(telemetry.get("total_spikes", 0))

            # Steering: turn angle based on descending motor neurons
            # Clamp steering change per step for smooth natural turns
            d_yaw = max(-30.0, min(30.0, turn * 18.0))
            yaw = (yaw + d_yaw) % 360.0

            # Forward propulsion
            current_speed = speed + max(0.0, forward * 0.5)
            rad = math.radians(yaw)
            dx = -math.sin(rad) * current_speed
            dz = math.cos(rad) * current_speed
            dy = 0.06 * math.sin(step * 0.8)  # Natural hovering flutter

            x += dx
            y += dy
            z += dz

            # Keep above ground level
            if y < -59.2:
                y = -58.5

            cmd_tp = f"tp @e[type=bee,limit=1] {x:.2f} {y:.2f} {z:.2f} {yaw:.1f} 0.0"
            cmd_hud = (
                f'title @a actionbar [{{"text":"[Brain Bee] ","color":"gold","bold":true}},'
                f'{{"text":"Spikes: {spikes:,} | ","color":"green"}},'
                f'{{"text":"Turn: {turn:+.1f} | ","color":"aqua"}},'
                f'{{"text":"Fwd: {forward:.3f}","color":"yellow"}}]'
            )

            waypoints.append((cmd_tp, cmd_hud, spikes, turn, forward))
            await asyncio.sleep(delay)

    return waypoints


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    results = asyncio.run(run_live_flight_session(steps=count, delay=0.05))
    print(f"Generated {len(results)} biological flight steps.")
    for tp, hud, spk, turn, fwd in results[:5]:
        print(f"Spikes={spk:5d} | Turn={turn:+.1f} | Command: {tp}")
