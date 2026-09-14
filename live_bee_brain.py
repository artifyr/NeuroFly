"""
Autonomous Live Minecraft Bee Connectome Controller.

Continuously runs in a background loop:
1. Queries the Bee's location, orientation, and surrounding Minecraft blocks.
2. Computes 3D spatial raycasts (distance to obstacles on the left and right).
3. Streams the raycasts to the FAFB FlyBrain SNN (running on your AMD Radeon RX 9070 XT).
4. Translates the Descending Motor spikes into continuous flight (turning, banking, flying forward).
5. Displays a live neural telemetry HUD in the Minecraft actionbar.
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


class LiveBeeBrainController:
    def __init__(self, ws_url: str = "ws://localhost:8765") -> None:
        self.ws_url = ws_url
        self.x = -51.08
        self.y = -59.0
        self.z = 57.49
        self.yaw = 15.0
        self.speed = 0.35

    def compute_flight_step(self, turn: float, forward: float) -> tuple[float, float, float, float]:
        """Convert SNN motor spikes into realistic continuous bee flight."""
        # Biological steering: turn is right_motor - left_motor
        # Scale steering response smoothly
        steering_delta = turn * 22.0
        steering_delta = max(-35.0, min(35.0, steering_delta))
        self.yaw = (self.yaw + steering_delta) % 360.0

        # Base forward flight speed + motor spike acceleration
        flight_speed = self.speed + forward * 0.8
        rad = math.radians(self.yaw)

        # Minecraft coordinates: dx = -sin(yaw), dz = cos(yaw)
        dx = -math.sin(rad) * flight_speed
        dz = math.cos(rad) * flight_speed

        # Gentle hovering elevation
        hover_dy = 0.08 * math.sin(time.time() * 2.5)

        self.x += dx
        self.y += hover_dy
        self.z += dz

        # Keep bee within hovering bounds above ground
        if self.y < -59.5:
            self.y = -59.0
        elif self.y > -56.0:
            self.y = -57.5

        return self.x, self.y, self.z, self.yaw
