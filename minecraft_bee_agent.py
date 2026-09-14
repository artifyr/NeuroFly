"""
Minecraft Bee Connectome Controller Bridge.

Bridges the real-time FAFB Spiking Neural Network WebSocket server (ws://localhost:8765)
with an in-game Minecraft Bee using the Minecraft MCP command interface.

Every tick:
1. Calculates raycasts / distance to obstacles in front-left and front-right of the Bee.
2. Sends sensory distances {"sensors": [left_dist, right_dist]} to FlyBrainSNN over WebSocket.
3. Receives motor commands {"turn": float, "forward": float, "total_spikes": int}.
4. Moves and rotates the Bee in Minecraft via tp / motion commands, displaying neural activity.
"""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from typing import Tuple

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


class MinecraftBeeAgent:
    """Represents the Minecraft Bee entity controlled by the SNN."""

    def __init__(self, tag: str = "fly_brain_agent") -> None:
        self.tag = tag
        self.x = -51.0
        self.y = -59.0
        self.z = 57.0
        self.yaw = 0.0  # degrees
        self.speed = 0.25

    def step_physics(self, turn: float, forward: float) -> Tuple[float, float, float, float]:
        """
        Update Bee's position and rotation based on biological motor outputs.
        - turn: Differential steering (positive = turn right, negative = turn left)
        - forward: Forward propulsion
        """
        # Turn sensitivity
        turn_angle = max(-30.0, min(30.0, turn * 15.0))
        self.yaw = (self.yaw + turn_angle) % 360.0

        # Movement in direction of yaw
        rad = math.radians(self.yaw)
        # Minecraft coordinate convention:
        # yaw 0 = +Z (South), 90 = -X (West), 180 = -Z (North), 270 = +X (East)
        dx = -math.sin(rad) * (forward * self.speed + 0.1)
        dz = math.cos(rad) * (forward * self.speed + 0.1)

        self.x += dx
        self.z += dz

        return self.x, self.y, self.z, self.yaw


def build_teleport_command(tag: str, x: float, y: float, z: float, yaw: float) -> str:
    """Build Minecraft command to update bee pose and render action title."""
    return f"tp @e[type=bee,tag={tag},limit=1] {x:.2f} {y:.2f} {z:.2f} {yaw:.1f} 0.0"


def build_status_actionbar(total_spikes: int, turn: float, forward: float) -> str:
    """Show live brain spikes HUD in Minecraft actionbar."""
    color = "green" if total_spikes > 1000 else "yellow"
    text = (
        f'{{"text":"[FAFB Brain Bee] ","color":"gold","bold":true}},'
        f'{{"text":"Spikes: {total_spikes:,} ","color":"{color}"}},'
        f'{{"text":"| Turn: {turn:+.2f} ","color":"aqua"}},'
        f'{{"text":"| Fwd: {forward:.3f}","color":"light_purple"}}'
    )
    return f'title @a actionbar [{text}]'
