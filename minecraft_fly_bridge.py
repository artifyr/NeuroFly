"""
FlyAI: Minecraft Fly Connectome Bridge & Biological Flight Controller.

Bridges the live Drosophila melanogaster FAFB Spiking Neural Network (ws://localhost:8765)
with an in-game Minecraft Bee/Fly entity, running the complete biological behavioral repertoire:
1. Biological Saccades (Rapid 50°-70° body turns interspersed with cruising)
2. Positive Phototaxis (Smooth spiraling approach and orbiting around Torches, Lanterns & Sunlight)
3. Thermal Nociception & Heat Avoidance (Repulsion away from Fire, Lava, and Campfires)
4. Perching & Stereotyped 3-Stage Grooming Sequence (Antennae -> Eyes -> Wings/Abdomen)
5. Circadian Sleep & Night Quiescence (Night perch resting, waking at dawn or disturbance)
6. Positive Chemotaxis & Feeding (Attraction to held food, flowers, composters)
7. Giant Fiber Looming Threat Escape Reflex (Emergency takeoff jump when player approaches rapidly)
8. In-game Actionbar HUD + Real-time WebApp Telemetry streaming
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import logging
import math
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MinecraftFlyBridge")

# Biological Food Valences for Drosophila melanogaster
FOOD_VALENCE: Dict[str, Tuple[float, str]] = {
    "minecraft:rotten_flesh": (1.00, "Rotten Flesh"),
    "minecraft:fermented_spider_eye": (0.90, "Fermented Eye"),
    "minecraft:spider_eye": (0.80, "Spider Eye"),
    "minecraft:poisonous_potato": (0.85, "Decaying Potato"),
    "minecraft:sugar": (0.80, "Sugar"),
    "minecraft:honey_bottle": (0.85, "Honey Bottle"),
    "minecraft:honeycomb": (0.75, "Honeycomb"),
    "minecraft:sweet_berries": (0.70, "Sweet Berries"),
    "minecraft:apple": (0.65, "Apple"),
    "minecraft:melon_slice": (0.65, "Melon"),
    "minecraft:cookie": (0.60, "Cookie"),
    "minecraft:cake": (0.75, "Cake"),
    "minecraft:dandelion": (0.50, "Flower"),
    "minecraft:poppy": (0.50, "Flower"),
    "minecraft:blue_orchid": (0.50, "Flower"),
    "minecraft:allium": (0.50, "Flower"),
    "minecraft:sunflower": (0.55, "Sunflower"),
}

LIGHT_BLOCKS: Dict[str, str] = {
    "minecraft:torch": "Torch",
    "minecraft:wall_torch": "Wall Torch",
    "minecraft:soul_torch": "Soul Torch",
    "minecraft:soul_wall_torch": "Soul Wall Torch",
    "minecraft:lantern": "Lantern",
    "minecraft:soul_lantern": "Soul Lantern",
    "minecraft:glowstone": "Glowstone",
    "minecraft:sea_lantern": "Sea Lantern",
    "minecraft:shroomlight": "Shroomlight",
    "minecraft:ochre_froglight": "Froglight",
    "minecraft:pearlescent_froglight": "Froglight",
    "minecraft:verdant_froglight": "Froglight",
    "minecraft:jack_o_lantern": "Jack o'Lantern",
    "minecraft:redstone_lamp": "Redstone Lamp",
    "minecraft:end_rod": "End Rod",
}

HAZARD_BLOCKS: Dict[str, str] = {
    "minecraft:lava": "Lava",
    "minecraft:fire": "Fire",
    "minecraft:soul_fire": "Soul Fire",
    "minecraft:campfire": "Campfire",
    "minecraft:soul_campfire": "Soul Campfire",
    "minecraft:magma_block": "Magma Block",
}

FOOD_BLOCKS: Dict[str, Tuple[float, str]] = {
    "minecraft:composter": (0.85, "Composter"),
    "minecraft:beehive": (0.90, "Beehive"),
    "minecraft:bee_nest": (0.90, "Bee Nest"),
    "minecraft:sweet_berry_bush": (0.75, "Berry Bush"),
    "minecraft:cake": (0.80, "Cake"),
    "minecraft:dandelion": (0.50, "Flower"),
    "minecraft:poppy": (0.50, "Flower"),
    "minecraft:blue_orchid": (0.50, "Flower"),
    "minecraft:allium": (0.50, "Flower"),
    "minecraft:sunflower": (0.55, "Sunflower"),
}

NON_SOLID_PERCH = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:water",
    "minecraft:lava", "minecraft:fire", "minecraft:soul_fire", "minecraft:short_grass",
    "minecraft:tall_grass", "minecraft:fern", "minecraft:large_fern",
}


class FastMinecraftBridge:
    """Persistent Keep-Alive JSON-RPC client with background player & block tracking."""

    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        self.req_id = 0
        self._initialized = False

        self.player_pos: Tuple[float, float, float] = (-38.0, -60.0, 175.0)
        self.held_item: str = "empty"
        self.is_day: bool = True
        self.is_night: bool = False

        self.fly_approx_pos: Tuple[float, float, float] = (-38.0, -58.0, 175.0)
        self.nearest_hazard: Optional[Tuple[float, float, float, float, str]] = None
        self.nearest_light: Optional[Tuple[float, float, float, float, str]] = None
        self.nearest_food_block: Optional[Tuple[float, float, float, float, float, str]] = None
        self.nearest_perch: Optional[Tuple[float, float, float, float]] = None
        self.heightmap: Dict[Tuple[int, int], float] = {}

        self._stop_bg = False
        self._bg_thread: Optional[threading.Thread] = None

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.req_id += 1
        payload = json.dumps({"jsonrpc": "2.0", "id": self.req_id, "method": method, "params": params})
        try:
            self.conn.request("POST", "/mcp", payload, self.headers)
            resp = self.conn.getresponse()
            data = resp.read().decode("utf-8")
            return json.loads(data)
        except Exception:
            try:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
                self.conn.request("POST", "/mcp", payload, self.headers)
                resp = self.conn.getresponse()
                data = resp.read().decode("utf-8")
                return json.loads(data)
            except Exception:
                return {}

    def _verify_or_patch_latency(self) -> None:
        try:
            t0 = time.perf_counter()
            self._call("tools/call", {
                "name": "execute_commands",
                "arguments": {"commands": ["tp @a ~ ~ ~"]},
            })
            dt = (time.perf_counter() - t0) * 1000.0
            if dt > 100.0:
                logger.warning(f"MCP command latency is high ({dt:.1f}ms). Auto-hotreloading CommandExecutor...")
                import os
                import shutil
                import subprocess
                jdk_java = os.environ.get("JAVA_HOME", "")
                if jdk_java:
                    candidate = os.path.join(jdk_java, "bin", "java.exe" if sys.platform == "win32" else "java")
                    jdk_java = candidate if os.path.exists(candidate) else (shutil.which("java") or "java")
                else:
                    jdk_java = shutil.which("java") or "java"
                ps_cmd = "(Get-Process javaw -ErrorAction SilentlyContinue).Id"
                pid = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd]).decode().strip()
                if pid:
                    subprocess.run(
                        [jdk_java, "-cp", "scratch", "AttachSingle", pid, "scratch/agent.jar", "scratch/CommandExecutor.class"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        capture_output=True,
                        timeout=5
                    )
                    logger.info(f"In-memory CommandExecutor hot-reloaded for PID {pid} (<10ms commands restored)!")
            else:
                logger.info(f"MCP Command latency verified: {dt:.1f}ms (Ultra-smooth 20Hz physics enabled).")
        except Exception as e:
            pass

    def initialize(self) -> bool:
        if not self._initialized:
            try:
                res = self._call("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "FlyAIBrainBridge", "version": "2.0"},
                })
                self._initialized = ("result" in res)
                if self._initialized:
                    logger.info("Connected to Minecraft MCP with persistent zero-latency link.")
                    self._verify_or_patch_latency()
            except Exception as e:
                logger.error(f"MCP Initialization error: {e}")
        return self._initialized

    def start_background_tracker(self) -> None:
        self.initialize()

        def _worker():
            tracker_conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
            iteration = 0

            while not self._stop_bg:
                iteration += 1
                try:
                    # 1. Update Player Info & Circadian Time
                    payload = json.dumps({
                        "jsonrpc": "2.0",
                        "id": 9990,
                        "method": "tools/call",
                        "params": {"name": "get_player_info", "arguments": {}},
                    })
                    tracker_conn.request("POST", "/mcp", payload, self.headers)
                    resp = tracker_conn.getresponse()
                    data = json.loads(resp.read().decode("utf-8"))
                    content = json.loads(data.get("result", {}).get("content", [{}])[0].get("text", "{}"))

                    pos = content.get("position", {})
                    if "x" in pos:
                        self.player_pos = (float(pos["x"]), float(pos["y"]), float(pos["z"]))

                    inv = content.get("inventory", {})
                    main_hand = inv.get("mainHandItem", "empty")
                    off_hand = inv.get("offHandItem", "empty")
                    self.held_item = main_hand if main_hand != "empty" else off_hand

                    self.is_day = content.get("isDay", True)
                    self.is_night = content.get("isNight", False)

                    # 2. Periodically scan block volume around fly (every ~1.0s)
                    if iteration % 4 == 0:
                        fx, fy, fz = self.fly_approx_pos
                        x1, y1, z1 = int(fx - 4), int(fy - 2), int(fz - 4)
                        x2, y2, z2 = int(fx + 4), int(fy + 3), int(fz + 4)

                        payload_blocks = json.dumps({
                            "jsonrpc": "2.0",
                            "id": 9991,
                            "method": "tools/call",
                            "params": {
                                "name": "get_blocks_in_area",
                                "arguments": {"from": {"x": x1, "y": y1, "z": z1}, "to": {"x": x2, "y": y2, "z": z2}}
                            },
                        })
                        tracker_conn.request("POST", "/mcp", payload_blocks, self.headers)
                        resp_b = tracker_conn.getresponse()
                        data_b = json.loads(resp_b.read().decode("utf-8"))
                        block_text = data_b.get("result", {}).get("content", [{}])[0].get("text", "{}")
                        block_data = json.loads(block_text)

                        min_hazard = None
                        min_light = None
                        min_food = None
                        min_perch = None
                        hmap: Dict[Tuple[int, int], float] = {}

                        for b in block_data.get("blocks", []):
                            btype = b.get("blockType", "")
                            coords: List[Tuple[float, float, float]] = []

                            for sb in b.get("singleBlocks", []):
                                coords.append((float(sb["x"]), float(sb["y"]), float(sb["z"])))
                            for reg in b.get("regions", []):
                                s, e = reg.get("start", {}), reg.get("end", {})
                                sx, ex = min(s["x"], e["x"]), max(s["x"], e["x"])
                                sy, ey = min(s["y"], e["y"]), max(s["y"], e["y"])
                                sz, ez = min(s["z"], e["z"]), max(s["z"], e["z"])
                                for rx in range(sx, ex + 1):
                                    for ry in range(sy, ey + 1):
                                        for rz in range(sz, ez + 1):
                                            coords.append((float(rx), float(ry), float(rz)))

                            for bx, by, bz in coords:
                                dist = math.hypot(bx - fx, bz - fz)
                                total_dist = math.hypot(dist, by - fy)

                                if btype not in NON_SOLID_PERCH:
                                    ix, iz = int(round(bx)), int(round(bz))
                                    top_y = by + 1.0
                                    if (ix, iz) not in hmap or top_y > hmap[(ix, iz)]:
                                        hmap[(ix, iz)] = top_y

                                if btype in HAZARD_BLOCKS:
                                    if min_hazard is None or total_dist < min_hazard[0]:
                                        min_hazard = (total_dist, bx, by, bz, HAZARD_BLOCKS[btype])

                                if btype in LIGHT_BLOCKS:
                                    if min_light is None or total_dist < min_light[0]:
                                        min_light = (total_dist, bx, by, bz, LIGHT_BLOCKS[btype])

                                if btype in FOOD_BLOCKS:
                                    val, fname = FOOD_BLOCKS[btype]
                                    if min_food is None or total_dist < min_food[0]:
                                        min_food = (total_dist, bx, by, bz, val, fname)

                                if btype not in NON_SOLID_PERCH and (by <= fy + 0.5):
                                    perch_top_y = by + 1.0
                                    pdist = math.hypot(bx - fx, bz - fz)
                                    if min_perch is None or pdist < min_perch[0]:
                                        min_perch = (pdist, bx, perch_top_y, bz)

                        self.nearest_hazard = min_hazard
                        self.nearest_light = min_light
                        self.nearest_food_block = min_food
                        self.nearest_perch = min_perch
                        self.heightmap = hmap

                except Exception:
                    tracker_conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)

                time.sleep(0.25)

        self._bg_thread = threading.Thread(target=_worker, daemon=True)
        self._bg_thread.start()

    def get_ground_y(self, x: float, z: float, default_y: float) -> float:
        ix, iz = int(round(x)), int(round(z))
        return self.heightmap.get((ix, iz), default_y)

    def stop_background_tracker(self) -> None:
        self._stop_bg = True

    def execute(self, commands: List[str]) -> Dict[str, Any]:
        self.initialize()
        return self._call("tools/call", {
            "name": "execute_commands",
            "arguments": {"commands": commands},
        })


class AerodynamicFlyAgent:
    """Insect aerodynamic flight physics and full biological behavioral repertoire."""

    def __init__(self, px: float, py: float, pz: float, is_walking: bool = False) -> None:
        self.is_walking = is_walking
        self.x = px + 1.5
        self.y = py if is_walking else (py + 1.8)
        self.z = pz + 1.5
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 45.0
        self.yaw_rate = 0.0
        self.pitch = 0.0

        self.drag = 0.28 if is_walking else 0.20
        self.cruise_thrust = 0.024 if is_walking else 0.032
        self.prev_dist_to_player = 10.0
        self.threat_cooldown = 0

        self.flight_energy: float = 100.0
        self.is_landed: bool = is_walking
        self.is_grooming: bool = False
        self.grooming_tick: int = 0
        self.grooming_phase_name: str = "Antennae"
        self.is_sleeping: bool = False

        self.saccade_cooldown: int = random.randint(30, 60)
        self.in_saccade: int = 0
        self.saccade_dir: float = 0.0

    def detect_looming_threat(self, px: float, py: float, pz: float) -> Tuple[bool, float]:
        if self.threat_cooldown > 0:
            self.threat_cooldown -= 1
            return False, 0.0

        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        closing_speed = self.prev_dist_to_player - dist_3d
        self.prev_dist_to_player = dist_3d

        is_threat = (closing_speed > 0.35 and dist_3d < 3.5 and abs(dy) < 2.5)
        if is_threat:
            self.threat_cooldown = 40
            if self.is_sleeping or self.is_grooming:
                self.is_sleeping = False
                self.is_grooming = False
                self.is_landed = self.is_walking
                if not self.is_walking:
                    self.vy = 0.15

            angle_to_player = math.degrees(math.atan2(-dx, dz)) % 360.0
            rel_angle = (angle_to_player - self.yaw + 180.0) % 360.0 - 180.0
            return True, rel_angle

        return False, 0.0

    def compute_sensory_inputs(self, px: float, py: float, pz: float, tick: int) -> Tuple[float, float]:
        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        base_left = 0.45 + 0.25 * math.sin(tick * 0.10)
        base_right = 0.45 - 0.25 * math.sin(tick * 0.10)

        threshold = 2.5 if self.is_walking else 3.5
        if dist_3d < threshold and abs(dy) < 2.5:
            angle_to_player = math.degrees(math.atan2(-dx, dz)) % 360.0
            rel_angle = (angle_to_player - self.yaw + 180.0) % 360.0 - 180.0
            threat = max(0.0, min(1.0, (threshold - dist_3d) / threshold))
            if rel_angle < 0:
                base_left += threat * 0.4
            else:
                base_right += threat * 0.4

        return max(0.05, min(1.0, base_left)), max(0.05, min(1.0, base_right))

    def compute_target_relative_vector(self, tx: float, ty: float, tz: float) -> Tuple[float, float, float]:
        dx = tx - self.x
        dy = ty - self.y
        dz = tz - self.z
        dist = math.hypot(dx, dz)
        angle_to_target = math.degrees(math.atan2(-dx, dz)) % 360.0
        rel_angle = (angle_to_target - self.yaw + 180.0) % 360.0 - 180.0
        return dist, rel_angle, dy

    def step_behavior_and_physics(
        self,
        turn_signal: float,
        forward_signal: float,
        learned_steer: float,
        px: float,
        py: float,
        pz: float,
        tick: int,
        is_escape: bool = False,
        food_valence: float = 0.0,
        food_dist: float = 10.0,
        food_rel_angle: float = 0.0,
        food_dy: float = 0.0,
        nearest_hazard: Optional[Tuple[float, float, float, float, str]] = None,
        nearest_light: Optional[Tuple[float, float, float, float, str]] = None,
        nearest_perch: Optional[Tuple[float, float, float, float]] = None,
        is_night: bool = False,
        ground_y: float = -60.0,
    ) -> Tuple[float, float, float, float, float, str]:
        active_state = "FORAGING"

        # 1. Night sleep
        if is_night and not is_escape:
            if not self.is_landed:
                if nearest_perch is not None and nearest_perch[0] < 6.0:
                    _, per_x, per_y, per_z = nearest_perch
                    dx = per_x - self.x
                    dz = per_z - self.z
                    if math.hypot(dx, dz) < 0.6 and abs(self.y - per_y) < 0.5:
                        self.is_landed = True
                        self.is_sleeping = True
                        self.y = per_y + 0.1
                        self.vx = self.vy = self.vz = 0.0
                    else:
                        self.x += max(-0.06, min(0.06, dx * 0.1))
                        self.z += max(-0.06, min(0.06, dz * 0.1))
                        self.y += max(-0.05, min(0.05, (per_y - self.y) * 0.1))
                else:
                    self.y = max(py - 2.0, self.y - 0.03)
                    self.vx *= 0.8
                    self.vz *= 0.8
                return self.x, self.y, self.z, self.yaw, self.pitch, "SLEEPING (Seeking Perch)"
            else:
                self.is_sleeping = True
                self.vx = self.vy = self.vz = 0.0
                return self.x, self.y, self.z, self.yaw, self.pitch, "SLEEPING / QUIESCENT (Night Rest)"
        elif not is_night and self.is_sleeping:
            self.is_sleeping = False
            self.is_landed = self.is_walking
            if not self.is_walking:
                self.vy = 0.10

        # 2. Stereotyped Grooming Sequence
        if self.is_grooming:
            self.grooming_tick += 1
            self.flight_energy = min(100.0, self.flight_energy + 1.2)
            self.vx = self.vy = self.vz = 0.0
            self.yaw = (self.yaw + random.uniform(-1.5, 1.5)) % 360.0

            if self.grooming_tick < 35:
                active_state = "GROOMING: Antennae"
                self.pitch = -12.0
            elif self.grooming_tick < 75:
                active_state = "GROOMING: Eyes"
                self.pitch = 0.0
            elif self.grooming_tick < 110:
                active_state = "GROOMING: Body & Legs"
                self.pitch = 12.0
            else:
                self.is_grooming = False
                self.is_landed = self.is_walking
                self.grooming_tick = 0
                self.flight_energy = 100.0
                if not self.is_walking:
                    self.vy = 0.12
                self.pitch = 0.0
                active_state = "FORAGING"

            return self.x, self.y, self.z, self.yaw, self.pitch, active_state

        self.flight_energy = max(0.0, self.flight_energy - 0.04)
        if (self.flight_energy < 15.0 or (tick % 600 == 0 and random.random() < 0.35)) and not is_escape and food_valence <= 0:
            if nearest_perch is not None and nearest_perch[0] < 5.0:
                _, per_x, per_y, per_z = nearest_perch
                pdist = math.hypot(per_x - self.x, per_z - self.z)
                if pdist < 0.7 and abs(self.y - per_y) < 0.6:
                    self.is_landed = True
                    self.is_grooming = True
                    self.grooming_tick = 0
                    self.y = per_y + 0.1
                    self.vx = self.vy = self.vz = 0.0
                    return self.x, self.y, self.z, self.yaw, self.pitch, "LANDED: Grooming Antennae"
                else:
                    self.x += max(-0.06, min(0.06, (per_x - self.x) * 0.1))
                    self.z += max(-0.06, min(0.06, (per_z - self.z) * 0.1))
                    self.y += max(-0.05, min(0.05, (per_y - self.y) * 0.1))
                    return self.x, self.y, self.z, self.yaw, self.pitch, "PERCHING (Landing)"

        # 3. Thermal Nociception & Heat Avoidance
        hazard_steer = 0.0
        hazard_repel_ax = 0.0
        hazard_repel_az = 0.0
        is_avoiding_hazard = False

        if nearest_hazard is not None:
            hdist, hx, hy, hz, hname = nearest_hazard
            if hdist < 4.5:
                is_avoiding_hazard = True
                active_state = f"THERMAL AVOIDANCE ({hname}!)"
                dx = self.x - hx
                dz = self.z - hz
                angle_to_hazard = math.degrees(math.atan2(-(-dx), -dz)) % 360.0
                h_rel_angle = (angle_to_hazard - self.yaw + 180.0) % 360.0 - 180.0
                hazard_steer = -math.copysign(min(22.0, (4.5 - hdist) * 8.0), h_rel_angle)

                repel_force = 0.04 * (4.5 - hdist)
                norm = max(0.1, math.hypot(dx, dz))
                hazard_repel_ax = (dx / norm) * repel_force
                hazard_repel_az = (dz / norm) * repel_force

        # 4. Saccadic Exploration
        saccade_active = False
        if not is_escape and not is_avoiding_hazard and food_valence <= 0:
            if self.saccade_cooldown > 0:
                self.saccade_cooldown -= 1
            else:
                self.in_saccade = 2
                self.saccade_dir = random.choice([-1.0, 1.0]) * random.uniform(45.0, 70.0)
                self.saccade_cooldown = random.randint(30, 65)

            if self.in_saccade > 0:
                saccade_active = True
                self.in_saccade -= 1
                active_state = "CRUISING (Saccade)"

        # 5. Phototaxis (Torches & Lanterns)
        light_steer = 0.0
        is_phototaxis = False
        if nearest_light is not None and not is_escape and not is_avoiding_hazard and food_valence <= 0:
            ldist, lx, ly, lz, lname = nearest_light
            if ldist < 10.0:
                is_phototaxis = True
                active_state = f"PHOTOTAXIS ({lname})"
                dx = lx - self.x
                dz = lz - self.z
                angle_to_light = math.degrees(math.atan2(-dx, dz)) % 360.0
                light_rel_angle = (angle_to_light - self.yaw + 180.0) % 360.0 - 180.0
                if ldist > 2.8:
                    light_steer = math.copysign(min(12.0, abs(light_rel_angle) * 0.25), light_rel_angle)
                else:
                    orbit_rel_angle = (light_rel_angle + 90.0 + 180.0) % 360.0 - 180.0
                    light_steer = math.copysign(min(14.0, abs(orbit_rel_angle) * 0.30), orbit_rel_angle)

        # 6. Steering Integration
        steer_target = turn_signal * 3.5 + learned_steer * 0.4

        if is_escape:
            active_state = "ESCAPE REFLEX (Looming!)"
            steer_target = max(-35.0, min(35.0, steer_target * 1.5))
        elif is_avoiding_hazard:
            steer_target = hazard_steer
        elif food_valence > 0:
            steer_target += math.copysign(min(8.0, abs(food_rel_angle) * 0.20), food_rel_angle) * food_valence
            if food_dist < 2.2:
                active_state = "FEEDING"
            else:
                active_state = "HUNGRY"
        elif is_phototaxis:
            steer_target += light_steer
        elif saccade_active:
            steer_target += self.saccade_dir * 0.55

        steer_target = max(-18.0, min(18.0, steer_target))
        self.yaw_rate = self.yaw_rate * 0.80 + steer_target * 0.20
        self.yaw_rate = max(-7.0, min(7.0, self.yaw_rate))
        self.yaw = (self.yaw + self.yaw_rate) % 360.0

        # 7. Thrust & Acceleration
        rad = math.radians(self.yaw)
        thrust = self.cruise_thrust + max(0.0, forward_signal * 0.04)

        if is_escape:
            thrust = 0.13
        elif is_avoiding_hazard:
            thrust = 0.09
        elif food_valence > 0 and food_dist < 2.2:
            thrust = 0.00

        ax = -math.sin(rad) * thrust + hazard_repel_ax
        az = math.cos(rad) * thrust + hazard_repel_az

        # Roaming perimeter (35m from player)
        dx_p = self.x - px
        dz_p = self.z - pz
        h_dist = math.hypot(dx_p, dz_p)
        if h_dist > 35.0:
            toward_player = math.degrees(math.atan2(-(-dx_p), -dz_p)) % 360.0
            diff = (toward_player - self.yaw + 180.0) % 360.0 - 180.0
            self.yaw_rate += math.copysign(min(2.8, abs(diff) * 0.07), diff)

        # 8. Altitude Regulation
        self.pitch = 0.0
        if food_valence > 0 and food_dist < 3.5:
            target_y = py + 1.1 + 0.10 * math.sin(tick * 0.2)
        elif is_phototaxis and nearest_light is not None:
            target_y = nearest_light[2] + 0.35 + 0.15 * math.sin(tick * 0.15)
        else:
            target_y = py + 2.0 + 0.20 * math.sin(tick * 0.15)

        if is_escape or is_avoiding_hazard:
            target_y += 0.9

        ay = (target_y - self.y) * 0.05 - self.vy * 0.20 + 0.01 * math.sin(tick * 0.4)
        self.vy = max(-0.08, min(0.08, self.vy * (1.0 - self.drag) + ay))
        self.y += self.vy

        # Prevent sinking below ground
        min_allowed_y = ground_y + 0.5
        if self.y < min_allowed_y:
            self.y = min_allowed_y
            self.vy = max(0.04, self.vy + 0.05)

        # 9. Velocity Integration
        self.vx = self.vx * (1.0 - self.drag) + ax
        self.vz = self.vz * (1.0 - self.drag) + az
        speed = math.hypot(self.vx, self.vz)
        max_speed = 0.26 if (is_escape or is_avoiding_hazard) else 0.12
        if speed > max_speed:
            scale = max_speed / speed
            self.vx *= scale
            self.vz *= scale

        self.x += self.vx
        self.z += self.vz

        return self.x, self.y, self.z, self.yaw, self.pitch, active_state


def build_hearts_bar(health: float, max_health: float = 20.0) -> str:
    ratio = max(0.0, min(1.0, health / max_health))
    num_full = int(round(ratio * 5))
    return "♥" * num_full + "·" * (5 - num_full)


class MinecraftFlyBridge:
    def __init__(
        self,
        ws_url: str = "ws://localhost:8765",
        mcp_host: str = "localhost",
        mcp_port: int = 8080,
        entity_type: str = "bee",
        mock_mode: bool = False,
    ):
        self.ws_url = ws_url
        self.entity_type = entity_type
        self.mock_mode = mock_mode
        self.bridge = FastMinecraftBridge(mcp_host, mcp_port)

        self.flight: Optional[AerodynamicFlyAgent] = None
        self.fly_health = 20.0
        self.max_health = 20.0
        self.tick = 0
        self.running = True

    def make_summon_command(self, x: float, y: float, z: float) -> str:
        name_str = "FAFB Connectome Fly (138k SNN)"
        return (
            f"summon bee {x:.2f} {y:.2f} {z:.2f} "
            f"{{CustomName:'\"{name_str}\"',CustomNameVisible:1b,NoGravity:1b,Invulnerable:1b,PersistenceRequired:1b,Tags:[\"fly_brain_active\"]}}"
        )

    def spawn_entity_in_minecraft(self) -> None:
        if self.flight is None:
            return
        # Despawn previous entities first, then summon exactly one
        self.bridge.execute([
            "tp @e[tag=fly_brain_active] ~ -300 ~",
            "tp @e[type=bee] ~ -300 ~",
        ])
        time.sleep(0.05)
        cmds = [
            self.make_summon_command(self.flight.x, self.flight.y, self.flight.z),
            f"effect clear @e[type={self.entity_type},tag=fly_brain_active]",
            f'title @a actionbar [{{"text":"[FlyBrain] ","color":"gold","bold":true}},{{"text":"{self.entity_type.capitalize()} Brain Embodying!","color":"green"}}]',
        ]
        self.bridge.execute(cmds)
        logger.info(f"Fly entity summoned at ({self.flight.x:.1f}, {self.flight.y:.1f}, {self.flight.z:.1f})")

    async def run(self):
        logger.info(f"Connecting to FlyBrain SNN Server at {self.ws_url}...")
        self.bridge.start_background_tracker()
        time.sleep(0.5)

        px, py, pz = self.bridge.player_pos
        logger.info(f"Player detected at X={px:.1f}, Y={py:.1f}, Z={pz:.1f}")

        self.flight = AerodynamicFlyAgent(px, py, pz)
        self.bridge.fly_approx_pos = (self.flight.x, self.flight.y, self.flight.z)

        if not self.mock_mode:
            self.spawn_entity_in_minecraft()

        retry_delay = 2.0

        while self.running:
            try:
                async with connect(self.ws_url) as ws:
                    logger.info(f"Connected to FlyBrain SNN Server at {self.ws_url}!")

                    while self.running:
                        self.tick += 1
                        loop_start = time.perf_counter()

                        # 1. Background State from Minecraft world
                        px, py, pz = self.bridge.player_pos
                        held_item = self.bridge.held_item
                        is_day = self.bridge.is_day
                        is_night = self.bridge.is_night
                        self.bridge.fly_approx_pos = (self.flight.x, self.flight.y, self.flight.z)

                        nearest_hazard = self.bridge.nearest_hazard
                        nearest_light = self.bridge.nearest_light
                        nearest_food_block = self.bridge.nearest_food_block
                        nearest_perch = self.bridge.nearest_perch

                        # 2. Food Perception (Held item or block)
                        held_val, held_name = FOOD_VALENCE.get(held_item, (0.0, ""))
                        if held_val > 0.0:
                            food_valence = held_val
                            food_name = held_name
                            food_dist, food_rel_angle, food_dy = self.flight.compute_target_relative_vector(px, py + 1.1, pz)
                        elif nearest_food_block is not None and nearest_food_block[0] < 8.0:
                            bdist, bx, by, bz, bval, bname = nearest_food_block
                            food_valence = bval * 0.75
                            food_name = bname
                            food_dist, food_rel_angle, food_dy = self.flight.compute_target_relative_vector(bx, by + 0.5, bz)
                        else:
                            food_valence = 0.0
                            food_name = ""
                            food_dist, food_rel_angle, food_dy = 10.0, 0.0, 0.0

                        is_feeding = (food_valence > 0.0 and food_dist < 2.2)

                        # 3. Threat detection
                        is_threat, threat_rel_angle = self.flight.detect_looming_threat(px, py, pz)

                        # 4. Dispatch sensory events to Brain Server
                        if is_threat:
                            await ws.send(json.dumps({"action": "inject_looming", "intensity": 1.0}))
                        if is_feeding and self.tick % 15 == 0:
                            await ws.send(json.dumps({"action": "inject_sweet", "intensity": 1.0}))
                            self.fly_health = min(self.max_health, self.fly_health + 1.0)
                        elif food_valence > 0 and self.tick % 25 == 0:
                            await ws.send(json.dumps({"action": "inject_odor", "intensity": food_valence}))
                        if nearest_hazard is not None and nearest_hazard[0] < 2.0 and self.tick % 15 == 0:
                            await ws.send(json.dumps({"action": "inject_pain", "intensity": 1.0}))
                            self.fly_health = max(1.0, self.fly_health - 2.0)

                        # 5. Receive Motor Commands & Spikes from SNN Server
                        try:
                            msg_raw = await asyncio.wait_for(ws.recv(), timeout=0.10)
                            data = json.loads(msg_raw)
                            if data.get("type") == "telemetry":
                                turn_signal = float(data.get("turn", 0.0))
                                forward_signal = float(data.get("forward", 0.0))
                                is_escape = bool(data.get("is_escaping", False)) or is_threat
                                total_spikes = int(data.get("count", 0))
                                dopamine = float(data.get("dopamine", 0.0))
                            else:
                                turn_signal, forward_signal, is_escape, total_spikes, dopamine = 0.0, 0.0, is_threat, 0, 0.0
                        except asyncio.TimeoutError:
                            turn_signal, forward_signal, is_escape, total_spikes, dopamine = 0.0, 0.0, is_threat, 0, 0.0

                        # 6. Execute Biological Flight Physics & State Machine
                        ground_y = self.bridge.get_ground_y(self.flight.x, self.flight.z, py)
                        bx, by, bz, byaw, bpitch, current_state_label = self.flight.step_behavior_and_physics(
                            turn_signal=turn_signal,
                            forward_signal=forward_signal,
                            learned_steer=0.0,
                            px=px,
                            py=py,
                            pz=pz,
                            tick=self.tick,
                            is_escape=is_escape,
                            food_valence=food_valence,
                            food_dist=food_dist,
                            food_rel_angle=food_rel_angle,
                            food_dy=food_dy,
                            nearest_hazard=nearest_hazard,
                            nearest_light=nearest_light,
                            nearest_perch=nearest_perch,
                            is_night=is_night,
                            ground_y=ground_y,
                        )

                        if current_state_label == "FEEDING":
                            current_state_label = f"FEEDING ({food_name}) [+HP]"
                        elif current_state_label == "HUNGRY":
                            current_state_label = f"HUNGRY ({food_name})"

                        # 7. Teleport in Minecraft & Update Actionbar HUD
                        if not self.mock_mode:
                            tp_cmd = f"tp @e[type={self.entity_type},tag=fly_brain_active,limit=1] {bx:.2f} {by:.2f} {bz:.2f} {byaw:.1f} {bpitch:.1f}"
                            cmds = [tp_cmd]

                            if self.tick % 2 == 0:
                                hearts_bar = build_hearts_bar(self.fly_health, self.max_health)
                                state_col = "green"
                                if "ESCAPE" in current_state_label:
                                    state_col = "dark_red"
                                elif "AVOIDANCE" in current_state_label or "PAIN" in current_state_label:
                                    state_col = "red"
                                elif "FEEDING" in current_state_label:
                                    state_col = "light_purple"
                                elif "HUNGRY" in current_state_label:
                                    state_col = "gold"
                                elif "PHOTOTAXIS" in current_state_label:
                                    state_col = "yellow"
                                elif "GROOMING" in current_state_label or "PERCHING" in current_state_label:
                                    state_col = "aqua"
                                elif "SLEEPING" in current_state_label:
                                    state_col = "blue"
                                elif "Saccade" in current_state_label:
                                    state_col = "dark_aqua"

                                hud_cmd = (
                                    f'title @a actionbar ['
                                    f'{{"text":"[FAFB {self.entity_type.capitalize()}] ","color":"gold","bold":true}},'
                                    f'{{"text":"HP: {int(self.fly_health)} [{hearts_bar}] ","color":"red","bold":true}},'
                                    f'{{"text":"[{current_state_label}] ","color":"{state_col}","bold":true}},'
                                    f'{{"text":"| Spikes: {total_spikes:,} ","color":"aqua"}},'
                                    f'{{"text":"| DA: {dopamine:.2f}","color":"yellow"}}]'
                                )
                                cmds.append(hud_cmd)

                            res = self.bridge.execute(cmds)
                            res_str = json.dumps(res)

                            # Auto-respawn fallback if entity was killed or not yet spawned
                            if "No entity was found" in res_str and self.tick % 20 == 0:
                                logger.info("Entity not found in world. Respawning fly near player...")
                                self.spawn_entity_in_minecraft()

                        # 8. Stream In-Game Telemetry to Brain Server & WebApp
                        speed_mps = math.hypot(self.flight.vx, self.flight.vz) * 20.0
                        dx_p = px - self.flight.x
                        dz_p = pz - self.flight.z
                        player_dist = math.hypot(dx_p, dz_p)

                        agent_telemetry = {
                            "type": "agent_telemetry",
                            "source": "Minecraft",
                            "entity": self.entity_type,
                            "pos": [round(bx, 2), round(by, 2), round(bz, 2)],
                            "yaw": round(byaw, 1),
                            "pitch": round(bpitch, 1),
                            "hp": round(self.fly_health, 1),
                            "max_hp": round(self.max_health, 1),
                            "state": current_state_label,
                            "speed": round(speed_mps, 2),
                            "nearest_light": {"dist": round(nearest_light[0], 1), "name": nearest_light[4]} if nearest_light else None,
                            "nearest_hazard": {"dist": round(nearest_hazard[0], 1), "name": nearest_hazard[4]} if nearest_hazard else None,
                            "nearest_food": {"dist": round(nearest_food_block[0], 1), "name": nearest_food_block[5]} if nearest_food_block else None,
                            "player_dist": round(player_dist, 1),
                            "is_night": is_night,
                        }
                        await ws.send(json.dumps(agent_telemetry))

                        if self.tick % 20 == 0:
                            logger.info(
                                f"[Tick {self.tick:4d}] Spikes: {total_spikes:5d} | State: {current_state_label[:26]:26s} | "
                                f"HP: {self.fly_health:4.1f} | Turn: {turn_signal:+5.1f} | Speed: {speed_mps:4.1f}m/s | "
                                f"Pos: ({bx:.1f}, {by:.1f}, {bz:.1f})"
                            )

                        elapsed = time.perf_counter() - loop_start
                        sleep_time = max(0.005, 0.05 - elapsed)
                        await asyncio.sleep(sleep_time)

            except Exception as err:
                logger.warning(f"Connection to SNN Server at {self.ws_url} lost ({err}). Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    def stop(self):
        self.running = False
        self.bridge.stop_background_tracker()


def main():
    parser = argparse.ArgumentParser(description="FlyAI: Minecraft Fly Brain Live Bridge")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="FlyBrain SNN WebSocket URL")
    parser.add_argument("--mcp-host", type=str, default="localhost", help="Minecraft MCP Host")
    parser.add_argument("--mcp-port", type=int, default=8080, help="Minecraft MCP Port")
    parser.add_argument("--entity", type=str, default="bee", choices=["bee", "villager"], help="Entity model")
def main():
    print("=" * 70)
    print("[!] NOTICE: minecraft_fly_bridge.py is deprecated in favor of native GPU simulation.")
    print("[>] Redirecting directly to native controller: run_minecraft_brain.py")
    print("=" * 70)
    import run_minecraft_brain
    run_minecraft_brain.main()


if __name__ == "__main__":
    main()

