"""
Live test loop: Run 20 ticks of Connectome-Driven Bee navigation in Minecraft.
"""

import asyncio
import json
import math
import random
from minecraft_bee_agent import MinecraftBeeAgent, build_teleport_command, build_status_actionbar
try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


async def run_fly_brain_minecraft_loop(ticks: int = 15):
    bee = MinecraftBeeAgent(tag="fly_brain_agent")
    commands_to_send = []

    print(f"Connecting to FlyBrain SNN WebSocket at ws://localhost:8765...")
    async with connect("ws://localhost:8765") as ws:
        print("Connected to FAFB Connectome SNN!")

        for t in range(ticks):
            # Modeled left / right eye optic lobe inputs
            # Simulate slight environmental sensory gradients (e.g. wall or light source)
            eye_left = 0.5 + 0.3 * math.sin(t * 0.4)
            eye_right = 0.5 - 0.3 * math.sin(t * 0.4)

            payload = {"sensors": [eye_left, eye_right]}
            await ws.send(json.dumps(payload))

            resp_raw = await ws.recv()
            telemetry = json.loads(resp_raw)

            turn = telemetry.get("turn", 0.0)
            forward = telemetry.get("forward", 0.0)
            total_spikes = telemetry.get("total_spikes", 0)

            # Update Bee physics
            x, y, z, yaw = bee.step_physics(turn, forward)
            cmd_tp = build_teleport_command("fly_brain_agent", x, y, z, yaw)
            cmd_hud = build_status_actionbar(total_spikes, turn, forward)
            commands_to_send.append((cmd_tp, cmd_hud, turn, forward, total_spikes))

    print(f"Computed {len(commands_to_send)} biological flight frames.")
    return commands_to_send


if __name__ == "__main__":
    result = asyncio.run(run_fly_brain_minecraft_loop(5))
    for tp, hud, turn, fwd, spk in result:
        print(f"Frame -> Turn: {turn:+.2f}, Forward: {fwd:.4f}, Spikes: {spk} | Command: {tp}")
