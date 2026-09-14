from __future__ import annotations

"""
FlyAI: Ultra-Smooth Biological Connectome Flight Controller for Minecraft.

Directly powers an in-game Minecraft Bee using the complete 138,584-neuron
Drosophila FAFB connectome spiking neural network on AMD Radeon RX 9070 XT (ROCm).

Features:
- Complete Drosophila Melanogaster Biological Behavioral Repertoire:
  1. Biological Saccades: Interspersed straight flights punctuated by rapid 50°-70° body saccades.
  2. Positive Phototaxis: Attraction and orbiting around Torches, Lanterns, Glowstone & Sunlight.
  3. Thermal Nociception & Heat Avoidance: Evasive repelling away from Fire, Lava, and Campfires.
  4. Perching & Stereotyped 3-Stage Grooming Sequence (Antennae -> Eyes -> Wings/Abdomen).
  5. Circadian Sleep & Night Quiescence: Restful perch during night, waking on disturbance/dawn.
  6. Positive Chemotaxis & Feeding: Odor detection of held food & organic blocks (Composter, Flowers).
  7. Giant Fiber Looming Threat & Pain Nociception: Fast escape reflex with dopamine memory.
- Dynamic In-Game Actionbar HUD:
  - Visual floating health meter (Hearts & Numerical HP).
  - Rich biological behavior states with color-coded alerts:
    [FORAGING], [CRUISING (Saccade)], [PHOTOTAXIS (Torch)], [HUNGRY (Item)], [FEEDING (+HP)],
    [THERMAL AVOIDANCE (Fire!)], [LANDED: Grooming], [SLEEPING / QUIESCENT], [ESCAPE REFLEX].
- Silent Micro-Stepped Insect Flight Physics (~6-10cm displacement per tick, zero teleport snapping).
- Background Asynchronous Player, Inventory & Block Area Tracking.
- Automatic Silent Respawn if entity despawns or is killed.
"""

import asyncio

import argparse
import http.client
import json
import math
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from connectome_loader import load_connectome_from_csv
from device_utils import get_device, get_device_info
from snn_engine import FlyBrainSNN

# Biological Food Valences for Drosophila melanogaster
FOOD_VALENCE: Dict[str, Tuple[float, str]] = {
    # 1. Decaying animal matter / Carrion (Highest attraction for flies)
    "minecraft:rotten_flesh": (1.00, "Rotten Flesh"),
    "minecraft:fermented_spider_eye": (0.90, "Fermented Eye"),
    "minecraft:spider_eye": (0.80, "Spider Eye"),
    "minecraft:poisonous_potato": (0.85, "Decaying Potato"),
    "minecraft:bone_meal": (0.50, "Bone Meal"),
    # 2. Sweet / Fermented Sugars
    "minecraft:sugar": (0.80, "Sugar"),
    "minecraft:honey_bottle": (0.85, "Honey Bottle"),
    "minecraft:honeycomb": (0.75, "Honeycomb"),
    "minecraft:sweet_berries": (0.70, "Sweet Berries"),
    "minecraft:apple": (0.65, "Apple"),
    "minecraft:melon_slice": (0.65, "Melon"),
    "minecraft:cookie": (0.60, "Cookie"),
    "minecraft:cake": (0.75, "Cake"),
    # 3. Organic Floral / Compost
    "minecraft:dandelion": (0.45, "Flower"),
    "minecraft:poppy": (0.45, "Flower"),
    "minecraft:flowering_azalea": (0.50, "Flowering Azalea"),
    "minecraft:wheat": (0.35, "Grain"),
}

# Environmental Light Sources (Phototaxis targets)
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

# Thermal Hazards (Thermonociceptive avoidance targets)
HAZARD_BLOCKS: Dict[str, str] = {
    "minecraft:lava": "Lava",
    "minecraft:fire": "Fire",
    "minecraft:soul_fire": "Soul Fire",
    "minecraft:campfire": "Campfire",
    "minecraft:soul_campfire": "Soul Campfire",
    "minecraft:magma_block": "Magma Block",
}

# Stationary Food Sources in World
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
    "minecraft:azure_bluet": (0.50, "Flower"),
    "minecraft:red_tulip": (0.50, "Flower"),
    "minecraft:orange_tulip": (0.50, "Flower"),
    "minecraft:white_tulip": (0.50, "Flower"),
    "minecraft:pink_tulip": (0.50, "Flower"),
    "minecraft:oxeye_daisy": (0.50, "Flower"),
    "minecraft:cornflower": (0.50, "Flower"),
    "minecraft:lily_of_the_valley": (0.50, "Flower"),
    "minecraft:sunflower": (0.55, "Sunflower"),
}

# Non-solid blocks to exclude from landing/perching
NON_SOLID_PERCH = {
    "minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:water",
    "minecraft:lava", "minecraft:fire", "minecraft:soul_fire", "minecraft:short_grass",
    "minecraft:tall_grass", "minecraft:fern", "minecraft:large_fern", "minecraft:vine",
    "minecraft:cave_vines", "minecraft:twisting_vines", "minecraft:weeping_vines",
    "minecraft:seagrass", "minecraft:tall_seagrass", "minecraft:kelp", "minecraft:kelp_plant",
}


class FastMinecraftBridge:
    """
    Persistent Keep-Alive JSON-RPC client with decoupled background player,
    inventory, time-of-day, and block environment tracking.
    """

    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        self.req_id = 0
        self._initialized = False

        # State tracked asynchronously
        self.player_pos: Tuple[float, float, float] = (-38.0, -60.0, 175.0)
        self.held_item: str = "empty"
        self.is_day: bool = True
        self.is_night: bool = False
        self.is_raining: bool = False

        # Environment scan results
        self.fly_approx_pos: Tuple[float, float, float] = (-38.0, -58.0, 175.0)
        self.nearest_hazard: Optional[Tuple[float, float, float, float, str]] = None  # (dist, x, y, z, name)
        self.nearest_light: Optional[Tuple[float, float, float, float, str]] = None   # (dist, x, y, z, name)
        self.nearest_food_block: Optional[Tuple[float, float, float, float, float, str]] = None  # (dist, x, y, z, val, name)
        self.nearest_perch: Optional[Tuple[float, float, float, float]] = None        # (dist, x, top_y, z)
        self.nearest_shelter: Optional[Tuple[float, float, float, float]] = None      # (dist, x, y, z)
        self.is_sheltered: bool = False
        self.known_shelters: Dict[Tuple[int, int, int], float] = {}  # (bx, by, bz) -> last_seen_time
        self.heightmap: Dict[Tuple[int, int], float] = {}

        self._stop_bg = False
        self._bg_thread: Optional[threading.Thread] = None

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.req_id += 1
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": self.req_id,
            "method": method,
            "params": params,
        })
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
            import os
            import shutil
            import subprocess
            ps_info = "(Get-Process javaw -ErrorAction SilentlyContinue | Select-Object -First 1 Id, Path | ConvertTo-Json)"
            ps_out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_info]).decode().strip()
            pid = None
            jdk_java = "java"
            if ps_out:
                try:
                    proc_info = json.loads(ps_out)
                    pid = str(proc_info.get("Id", ""))
                    javaw_path = proc_info.get("Path", "")
                    if javaw_path:
                        cand = os.path.join(os.path.dirname(javaw_path), "java.exe" if sys.platform == "win32" else "java")
                        if os.path.exists(cand):
                            jdk_java = cand
                except Exception:
                    pass

            if jdk_java == "java":
                jdk_env = os.environ.get("JAVA_HOME", "")
                if jdk_env:
                    cand = os.path.join(jdk_env, "bin", "java.exe" if sys.platform == "win32" else "java")
                    if os.path.exists(cand):
                        jdk_java = cand
                else:
                    jdk_java = shutil.which("java") or "java"

            if dt > 100.0:
                print(f"[!] Warning: MCP command latency is high ({dt:.1f}ms). Auto-hotreloading CommandExecutor...")
                if pid:
                    subprocess.run(
                        [jdk_java, "-cp", "scratch", "AttachSingle", pid, "scratch/agent.jar", "scratch/CommandExecutor.class"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        capture_output=True,
                        timeout=5
                    )
                    print(f"[OK] In-memory CommandExecutor hot-reloaded for PID {pid} (<10ms commands restored)!")
            else:
                print(f"[OK] MCP Command latency verified: {dt:.1f}ms (Ultra-smooth 20Hz physics enabled).")

            # Verify rain & weather reporting in PlayerInfoProvider
            info_res = self._call("tools/call", {"name": "get_player_info", "arguments": {}})
            content_str = info_res.get("result", {}).get("content", [{}])[0].get("text", "{}")
            if content_str and "isRaining" not in content_str:
                print("[!] In-memory PlayerInfoProvider lacks 'isRaining'. Auto-patching live JVM...")
                if pid:
                    subprocess.run(
                        [jdk_java, "-cp", "scratch", "Attach", pid, "scratch/live_agent.jar", "cuspymd.mcp.mod.utils.PlayerInfoProvider", "scratch/out/cuspymd/mcp/mod/utils/PlayerInfoProvider.class"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        capture_output=True,
                        timeout=5
                    )
                    print(f"[OK] In-memory PlayerInfoProvider hot-reloaded for PID {pid} (Live rain detection enabled)!")
        except Exception as e:
            pass

    def initialize(self) -> None:
        if not self._initialized:
            try:
                self._call("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "FlyAIBrainFast", "version": "2.0"},
                })
                self._initialized = True
                print("[OK] Connected to Minecraft MCP with persistent zero-latency link.")
                self._verify_or_patch_latency()
            except Exception as e:
                print(f"[!] MCP Initialization error: {e}")

    def start_background_tracker(self) -> None:
        """Runs player and environment block tracking asynchronously."""
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
                    if "isRaining" in content:
                        self.is_raining = bool(content["isRaining"])

                    # 2. Periodically scan block volume around fly (every ~1.0s)
                    if iteration % 4 == 0:
                        fx, fy, fz = self.fly_approx_pos

                        # Determine scan boxes adhering strictly to MCP limit (<= 10 blocks per axis!)
                        # We perform a dual-box scan: Ground-to-low-canopy (Box 1) and High-canopy (Box 2)
                        # to ensure full tree heights (oak, spruce, birch, mangrove up to 16 blocks high) are discovered.
                        scan_boxes = [
                            # Box 1: Immediate surroundings & low canopy
                            (int(round(fx - 4)), int(round(fy - 2)), int(round(fz - 4)),
                             int(round(fx + 4)), int(round(fy + 7)), int(round(fz + 4))),
                            # Box 2: High canopy directly overhead
                            (int(round(fx - 4)), int(round(fy + 7)), int(round(fz - 4)),
                             int(round(fx + 4)), int(round(fy + 16)), int(round(fz + 4))),
                        ]

                        # In rain, also alternate scanning horizontal offset sectors to discover distant trees
                        if self.is_raining:
                            quad = (iteration // 4) % 4
                            ox = 6 if quad == 0 else (-6 if quad == 1 else 0)
                            oz = 6 if quad == 2 else (-6 if quad == 3 else 0)
                            scan_boxes.append((
                                int(round(fx + ox - 4)), int(round(fy - 2)), int(round(fz + oz - 4)),
                                int(round(fx + ox + 4)), int(round(fy + 7)), int(round(fz + oz + 4)),
                            ))

                        min_hazard = None
                        min_light = None
                        min_food = None
                        min_perch = None

                        for (x1, y1, z1, x2, y2, z2) in scan_boxes:
                            # Safety clamp to guarantee strictly <= 10 blocks per axis
                            if x2 - x1 >= 10:
                                x2 = x1 + 9
                            if y2 - y1 >= 10:
                                y2 = y1 + 9
                            if z2 - z1 >= 10:
                                z2 = z1 + 9

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

                            # Parse blocks and build ground heightmap & canopy memory
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
                                    # Ground surface tracking: only terrain blocks (exclude elevated canopy leaves, logs, or roofs)
                                    is_elevated_structure = (
                                        btype.endswith("_leaves") or
                                        btype.endswith("_sapling") or
                                        btype.endswith("_propagule") or
                                        btype.endswith("_vine") or
                                        btype.endswith("_vines") or
                                        (by > fy + 1.5)  # Any block >1.5m above fly is overhead canopy, NOT ground!
                                    )
                                    if btype not in NON_SOLID_PERCH and not is_elevated_structure:
                                        ix, iz = int(round(bx)), int(round(bz))
                                        top_y = by + 1.0
                                        if (ix, iz) not in self.heightmap or top_y > self.heightmap[(ix, iz)]:
                                            self.heightmap[(ix, iz)] = top_y

                                    # Canopy / shelter detection: tree leaves, logs, wood, roofs, building blocks
                                    is_canopy = (
                                        btype.endswith("_leaves") or
                                        btype.endswith("_log") or
                                        btype.endswith("_wood") or
                                        btype.endswith("_planks") or
                                        btype.endswith("_slab") or
                                        btype.endswith("_stairs") or
                                        "roof" in btype or
                                        (btype not in NON_SOLID_PERCH and not btype.endswith("_flower") and by >= fy + 0.3)
                                    )
                                    if is_canopy:
                                        self.known_shelters[(int(round(bx)), int(round(by)), int(round(bz)))] = time.time()

                                    # Thermal hazards
                                    if btype in HAZARD_BLOCKS:
                                        if min_hazard is None or total_dist < min_hazard[0]:
                                            min_hazard = (total_dist, bx, by, bz, HAZARD_BLOCKS[btype])

                                    # Light sources
                                    if btype in LIGHT_BLOCKS:
                                        if min_light is None or total_dist < min_light[0]:
                                            min_light = (total_dist, bx, by, bz, LIGHT_BLOCKS[btype])

                                    # World food blocks
                                    if btype in FOOD_BLOCKS:
                                        val, fname = FOOD_BLOCKS[btype]
                                        if min_food is None or total_dist < min_food[0]:
                                            min_food = (total_dist, bx, by, bz, val, fname)

                                    # Solid perch candidate (top surface at by + 1.0, exclude leaves so fly doesn't perch on top of tree)
                                    if btype not in NON_SOLID_PERCH and not btype.endswith("_leaves") and (by <= fy + 0.5):
                                        perch_top_y = by + 1.0
                                        pdist = math.hypot(bx - fx, bz - fz)
                                        if min_perch is None or pdist < min_perch[0]:
                                            min_perch = (pdist, bx, perch_top_y, bz)

                        # Clean up known_shelters that are too distant (> 45m) or very stale (> 600s)
                        now = time.time()
                        stale_keys = [k for k, t in self.known_shelters.items() if (now - t > 600.0 or math.hypot(k[0] - fx, k[2] - fz) > 45.0)]
                        for k in stale_keys:
                            self.known_shelters.pop(k, None)

                        # Group canopy blocks by horizontal column (bx, bz)
                        col_canopies: Dict[Tuple[int, int], List[int]] = {}
                        for (sx, sy, sz) in self.known_shelters.keys():
                            col_canopies.setdefault((sx, sz), []).append(sy)

                        is_sheltered = False
                        min_shelter = None
                        overhead_min_y = None

                        # Check if fly is under any canopy column within 1.85m horizontal distance
                        for (sx, sz), y_list in col_canopies.items():
                            pdist = math.hypot(sx - fx, sz - fz)
                            # Overhead canopy blocks in this column
                            overhead_in_col = [y for y in y_list if y >= fy + 0.15]
                            if pdist <= 1.85 and overhead_in_col:
                                is_sheltered = True
                                lowest_overhead = min(overhead_in_col)
                                if overhead_min_y is None or lowest_overhead < overhead_min_y:
                                    overhead_min_y = lowest_overhead

                        # Compute nearest shelter target: targets underside of nearest canopy
                        for (sx, sz), y_list in col_canopies.items():
                            pdist = math.hypot(sx - fx, sz - fz)
                            lowest_y = min(y_list)
                            ground_at_col = self.get_ground_y(sx, sz, fy - 1.5)
                            target_canopy_y = min(lowest_y - 0.6, max(ground_at_col + 0.4, lowest_y - 0.9))
                            if min_shelter is None or pdist < min_shelter[0]:
                                min_shelter = (pdist, float(sx), float(target_canopy_y), float(sz))

                        self.nearest_hazard = min_hazard
                        self.nearest_light = min_light
                        self.nearest_food_block = min_food
                        self.nearest_perch = min_perch
                        self.nearest_shelter = min_shelter
                        self.is_sheltered = is_sheltered

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

    def execute(self, commands: List[str], validate_safety: bool = False) -> Dict[str, Any]:
        self.initialize()
        return self._call("tools/call", {
            "name": "execute_commands",
            "arguments": {"commands": commands, "validate_safety": validate_safety},
        })


class AerodynamicFlyAgent:
    """
    Simulates continuous insect aerodynamic flight physics and natural fly behaviors:
    - Mass, continuous velocity vector (vx, vy, vz).
    - Biological Saccadic exploration: 50-70 degree rapid turns every 1.5-3.0s.
    - Phototaxis: Smooth orbiting and approach toward torches and lanterns.
    - Thermal Nociception: Sharp repulsive evasion from fire and lava hazards.
    - Perching & Stereotyped 3-Stage Grooming Sequence (Antennae, Eyes, Wings).
    - Circadian Sleep & Quiescence: Settles on perch at night, awakening on disturbance.
    - Chemotaxis & Proboscis Extension Reflex (PER) feeding.
    - Giant Fiber Looming Threat Escape Reflex.
    """

    def __init__(self, px: float, py: float, pz: float, is_walking: bool = False) -> None:
        self.is_walking = is_walking
        self.x = px + 2.0
        self.y = py if is_walking else (py + 2.0)
        self.z = pz + 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 45.0
        self.yaw_rate = 0.0
        self.pitch = 0.0

        # Locomotion constants
        self.drag = 0.28 if is_walking else 0.20
        self.cruise_thrust = 0.024 if is_walking else 0.032
        self.prev_dist_to_player = 10.0
        self.threat_cooldown = 0

        # Biological State Variables
        self.flight_energy: float = 100.0  # Depletes during locomotion, recharges while grooming
        self.is_landed: bool = is_walking
        self.is_grooming: bool = False
        self.grooming_tick: int = 0
        self.grooming_phase_name: str = "Antennae"
        self.is_sleeping: bool = False
        self.in_rear_blind_spot: bool = False
        self.annoyed_jump_ticks: int = 0
        self.annoyed_cooldown: int = 0
        self.annoyed_base_x: float = self.x
        self.annoyed_base_y: float = self.y
        self.annoyed_base_z: float = self.z

        # Saccade state
        self.saccade_cooldown: int = random.randint(30, 60)
        self.in_saccade: int = 0
        self.saccade_dir: float = 0.0

    def detect_looming_threat(self, px: float, py: float, pz: float) -> Tuple[bool, float]:
        """
        Biological Looming Threat Detector (Giant Fiber Reflex):
        Calculates 3D Euclidean distance, closing speed, and compound eye visual field.
        Real Drosophila have a posterior physical blind spot covering roughly 1/6th
        of their surroundings directly behind them (~60° cone, |rel_angle| > 150°).
        Approaching from directly behind does NOT trigger the looming escape reflex!
        """
        if self.threat_cooldown > 0:
            self.threat_cooldown -= 1
            return False, 0.0

        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        closing_speed = self.prev_dist_to_player - dist_3d
        self.prev_dist_to_player = dist_3d

        angle_to_player = math.degrees(math.atan2(-dx, dz)) % 360.0
        rel_angle = (angle_to_player - self.yaw + 180.0) % 360.0 - 180.0

        # Physical rear blind spot covers ~1/6th (60°) directly behind (|rel_angle| > 150°)
        self.in_rear_blind_spot = (abs(rel_angle) > 150.0 and dist_3d < 4.0)

        # Looming reflex triggers only if closing fast, nearby, vertically aligned, and NOT in blind spot
        is_threat = (closing_speed > 0.35 and dist_3d < 3.5 and abs(dy) < 2.5 and not self.in_rear_blind_spot)
        if is_threat:
            self.threat_cooldown = 40  # ~2.0s refractory period
            # If sleeping or grooming, immediately abort and wake up!
            if self.is_sleeping or self.is_grooming:
                self.is_sleeping = False
                self.is_grooming = False
                self.is_landed = self.is_walking
                if not self.is_walking:
                    self.vy = 0.15

            return True, rel_angle

        return False, 0.0

    def detect_sound_or_touch(self, px: float, py: float, pz: float) -> bool:
        """
        Johnston's Organ & Chordotonal Mechanoreceptors (Sound as Physical Touch):
        In fruit flies, acoustic vibrations (e.g. footsteps, breaking blocks, note blocks)
        within 1.5 blocks are perceived as tactile physical touch, triggering an
        annoyed flinch hop, smoke annoyance puff, and startle flight response.
        """
        if self.annoyed_cooldown > 0:
            self.annoyed_cooldown -= 1

        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist_3d <= 1.5:
            if self.annoyed_jump_ticks <= 0 and self.annoyed_cooldown <= 0:
                self.annoyed_jump_ticks = 16  # ~0.8s of rapid up-down annoyance oscillation
                self.annoyed_base_x = self.x
                self.annoyed_base_y = self.y
                self.annoyed_base_z = self.z
                self.annoyed_cooldown = 40    # 2.0s cooldown before next annoyance trigger
                if self.is_sleeping or self.is_grooming:
                    self.is_sleeping = False
                    self.is_grooming = False
                    self.is_landed = self.is_walking
            return True
        return False

    def compute_sensory_inputs(self, px: float, py: float, pz: float, tick: int) -> Tuple[float, float]:
        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        base_left = 0.45 + 0.25 * math.sin(tick * 0.10)
        base_right = 0.45 - 0.25 * math.sin(tick * 0.10)

        # Obstacle avoidance from player if close in 3D space
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
        food_name: str = "",
        nearest_hazard: Optional[Tuple[float, float, float, float, str]] = None,
        nearest_light: Optional[Tuple[float, float, float, float, str]] = None,
        nearest_perch: Optional[Tuple[float, float, float, float]] = None,
        nearest_shelter: Optional[Tuple[float, float, float, float]] = None,
        is_night: bool = False,
        is_raining: bool = False,
        is_sheltered: bool = False,
        ground_y: float = -60.0,
    ) -> Tuple[float, float, float, float, float, str]:
        """
        Executes full biological flight/walking dynamics & behavior state machine:
        - Rain Avoidance: Rains are harmful; seeks overhead cover or shelters under ceiling.
        - Johnston's Organ Annoyance: Sounds within 1.5m trigger tactile flinch hop & evasive flight.
        - Internal Energy Pool: Drains during flight; <50% triggers hunger dominance & odor tracking.
        - 60° Posterior Blind Spot: Compounds eyes blind directly behind (|rel_angle| > 150°).
        - Phototaxis, Thermonociception, Saccadic Cruise, Circadian Quiescence, Grooming.
        Returns: (x, y, z, yaw, pitch, active_state_name)
        """
        active_state = "FORAGING"

        # Deplete internal metabolic energy pool
        if self.is_walking:
            self.flight_energy = max(0.0, self.flight_energy - 0.025)
        else:
            self.flight_energy = max(0.0, self.flight_energy - 0.05)

        is_hungry = (self.flight_energy < 50.0)

        # -------------------------------------------------------------
        # 1. NIGHTTIME CIRCADIAN QUIESCENCE / SLEEP
        # -------------------------------------------------------------
        if is_night and not is_escape and self.annoyed_jump_ticks <= 0:
            if self.is_walking:
                self.is_sleeping = True
                self.vx = self.vy = self.vz = 0.0
                self.y = ground_y
                self.pitch = 10.0
                active_state = "SLEEPING / QUIESCENT (Night Rest)"
                return self.x, self.y, self.z, self.yaw, self.pitch, active_state

            # Flying mode perching and sleep
            if not self.is_landed:
                if nearest_perch is not None and nearest_perch[0] < 6.0:
                    _, per_x, per_y, per_z = nearest_perch
                    dx = per_x - self.x
                    dz = per_z - self.z
                    pdist = math.hypot(dx, dz)
                    if pdist < 0.6 and abs(self.y - per_y) < 0.5:
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
                    if self.y <= ground_y + 0.15:
                        self.is_landed = True
                        self.is_sleeping = True
                        self.y = ground_y
                        self.vx = self.vy = self.vz = 0.0
                        active_state = "SLEEPING / QUIESCENT (Night Rest)"
                        return self.x, self.y, self.z, self.yaw, self.pitch, active_state
                active_state = "SLEEPING (Seeking Perch)"
                return self.x, self.y, self.z, self.yaw, self.pitch, active_state
            else:
                self.is_sleeping = True
                self.vx = self.vy = self.vz = 0.0
                active_state = "SLEEPING / QUIESCENT (Night Rest)"
                return self.x, self.y, self.z, self.yaw, self.pitch, active_state
        elif not is_night and self.is_sleeping:
            self.is_sleeping = False
            self.is_landed = self.is_walking
            if not self.is_walking:
                self.vy = 0.10
            self.pitch = 0.0

        # -------------------------------------------------------------
        # 2. RAIN AVOIDANCE & SHELTER SEEKING (Lethal rain droplet evasion)
        # -------------------------------------------------------------
        shelter_steer = 0.0
        is_seeking_shelter = False
        if is_raining and not is_escape:
            if not is_sheltered:
                is_seeking_shelter = True
                active_state = "SEEKING SHELTER (Rain Pain!)"
                if nearest_shelter is not None:
                    _, sx, sy, sz = nearest_shelter
                    dx = sx - self.x
                    dz = sz - self.z
                    angle_to_shelter = math.degrees(math.atan2(-dx, dz)) % 360.0
                    shelter_rel = (angle_to_shelter - self.yaw + 180.0) % 360.0 - 180.0
                    shelter_steer = math.copysign(min(24.0, abs(shelter_rel) * 0.45), shelter_rel)
                    # Fly towards shelter altitude (stay underneath canopy)
                    target_y = max(ground_y + 0.3, min(self.y + 0.08, sy))
                    self.y += max(-0.08, min(0.08, (target_y - self.y) * 0.25))
                else:
                    # In open field without known shelter: stay low and explore
                    shelter_steer = random.uniform(-6.0, 6.0)
                    self.y = max(ground_y + 0.6, self.y - 0.04)
            else:
                # Under safe overhead cover (tree leaf canopy, cave roof, or building ceiling)
                active_state = "SHELTERED (Safe from Rain)"
                self.vx *= 0.20
                self.vz *= 0.20
                self.vy = 0.0
                self.pitch = 2.0
                if nearest_shelter is not None:
                    self.y = max(ground_y + 0.25, min(self.y, nearest_shelter[2]))
                if not self.is_grooming:
                    self.is_grooming = True
                    self.grooming_tick = 0

        # -------------------------------------------------------------
        # 3. LANDED / STANDING STEREOTYPED 3-STAGE GROOMING
        # -------------------------------------------------------------
        if self.is_grooming:
            self.grooming_tick += 1
            self.flight_energy = min(100.0, self.flight_energy + 1.5)
            self.vx = self.vy = self.vz = 0.0
            if self.is_walking:
                self.y = ground_y
            elif is_raining and is_sheltered and nearest_shelter is not None:
                safe_canopy_y = nearest_shelter[2]
                self.y = max(ground_y + 0.20, min(self.y, safe_canopy_y))

            # Small micro-yaw grooming jitters
            self.yaw = (self.yaw + random.uniform(-1.5, 1.5)) % 360.0

            prefix = "SHELTERED: " if (is_raining and is_sheltered) else "GROOMING: "
            if self.grooming_tick < 35:
                self.grooming_phase_name = "Antennae"
                active_state = prefix + "Antennae"
                self.pitch = -12.0
            elif self.grooming_tick < 75:
                self.grooming_phase_name = "Eyes & Proboscis"
                active_state = prefix + "Eyes"
                self.pitch = 0.0
            elif self.grooming_tick < 110:
                self.grooming_phase_name = "Wings & Abdomen"
                active_state = prefix + "Body & Legs"
                self.pitch = 12.0
            else:
                # Finished grooming
                self.is_grooming = False
                self.is_landed = self.is_walking
                self.grooming_tick = 0
                self.flight_energy = 100.0
                if not self.is_walking:
                    self.vy = 0.0 if (is_raining and is_sheltered) else 0.08
                self.pitch = 0.0
                active_state = "SHELTERED (Safe from Rain)" if (is_raining and is_sheltered) else "FORAGING"

            return self.x, self.y, self.z, self.yaw, self.pitch, active_state

        # Check if needs grooming (energy low < 15.0 or random pause) when not in urgent state
        if (self.flight_energy < 15.0 or (tick % 600 == 0 and random.random() < 0.35)) and not is_escape and food_valence <= 0 and not is_seeking_shelter and not (is_raining and not is_sheltered) and self.annoyed_jump_ticks <= 0:
            if self.is_walking:
                self.is_grooming = True
                self.grooming_tick = 0
                self.vx = self.vy = self.vz = 0.0
                return self.x, self.y, self.z, self.yaw, self.pitch, "GROOMING: Antennae"
            elif nearest_perch is not None and nearest_perch[0] < 5.0 and not is_raining:
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

        # -------------------------------------------------------------
        # 4. THERMAL NOCICEPTION & AVOIDANCE (Fire, Lava, Campfires)
        # -------------------------------------------------------------
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

                turn_away = -math.copysign(min(22.0, (4.5 - hdist) * 8.0), h_rel_angle)
                hazard_steer = turn_away

                repel_force = 0.04 * (4.5 - hdist)
                norm = max(0.1, math.hypot(dx, dz))
                hazard_repel_ax = (dx / norm) * repel_force
                hazard_repel_az = (dz / norm) * repel_force

        # -------------------------------------------------------------
        # 5. SACCADIC EXPLORATION (Disabled if hungry, annoyed or seeking shelter)
        # -------------------------------------------------------------
        saccade_active = False
        if not is_escape and not is_avoiding_hazard and food_valence <= 0 and not is_hungry and not is_seeking_shelter and self.annoyed_jump_ticks <= 0:
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

        # -------------------------------------------------------------
        # 6. PHOTOTAXIS (Attraction & Orbiting Light Sources)
        # -------------------------------------------------------------
        light_steer = 0.0
        is_phototaxis = False
        if nearest_light is not None and not is_escape and not is_avoiding_hazard and food_valence <= 0 and not is_seeking_shelter and self.annoyed_jump_ticks <= 0:
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

        # -------------------------------------------------------------
        # 7. STEERING INTEGRATION
        # -------------------------------------------------------------
        steer_target = turn_signal * 3.5 + learned_steer * 0.4

        if is_escape:
            active_state = "ESCAPE REFLEX (Pain!)"
            steer_target = max(-35.0, min(35.0, steer_target * 1.5))
        elif is_avoiding_hazard:
            steer_target = hazard_steer
        elif is_seeking_shelter:
            steer_target = shelter_steer + learned_steer * 0.5
        elif self.annoyed_jump_ticks > 0:
            active_state = "ANNOYED (Sound/Touch!)"
            steer_target = random.uniform(-4.0, 4.0)
        elif is_hungry or food_valence > 0:
            # Energy < 50% shifts dominant behavior to actively tracking sugar/melon odor vectors!
            hunger_boost = 1.0 + (50.0 - min(50.0, self.flight_energy)) / 25.0
            if food_valence > 0:
                steer_target += math.copysign(min(14.0, abs(food_rel_angle) * 0.30), food_rel_angle) * food_valence * hunger_boost
                if food_dist < 2.2:
                    active_state = "FEEDING"
                    self.flight_energy = min(100.0, self.flight_energy + 1.5)  # Recharging energy!
                else:
                    active_state = "HUNGRY"
            else:
                active_state = "HUNGRY"
        elif is_phototaxis:
            steer_target += light_steer
        elif saccade_active:
            steer_target += self.saccade_dir * 0.55

        steer_target = max(-18.0, min(18.0, steer_target))

        # Heavy damping for silky smooth curved banking
        self.yaw_rate = self.yaw_rate * 0.80 + steer_target * 0.20
        self.yaw_rate = max(-7.0, min(7.0, self.yaw_rate))
        self.yaw = (self.yaw + self.yaw_rate) % 360.0

        # -------------------------------------------------------------
        # 8. THRUST & ACCELERATION (Walking vs Flying)
        # -------------------------------------------------------------
        rad = math.radians(self.yaw)
        thrust = self.cruise_thrust + max(0.0, forward_signal * (0.025 if self.is_walking else 0.04))

        if is_escape:
            thrust = 0.09 if self.is_walking else 0.13
        elif is_avoiding_hazard or is_seeking_shelter:
            thrust = 0.07 if self.is_walking else 0.10
        elif self.annoyed_jump_ticks > 0:
            thrust = 0.00  # Stays strictly in place during annoyance hop
        elif food_valence > 0 and food_dist < (1.4 if self.is_walking else 2.2):
            thrust = 0.00  # Stopped in front of food/player to feed!

        ax = -math.sin(rad) * thrust + hazard_repel_ax
        az = math.cos(rad) * thrust + hazard_repel_az

        # Sound/touch annoyance: move up and down in annoyance in the same place
        was_annoyed = (self.annoyed_jump_ticks > 0)
        if was_annoyed:
            self.annoyed_jump_ticks -= 1
            # Maintain position in the same place horizontally (zero lateral drift)
            self.vx *= 0.15
            self.vz *= 0.15
            ax = 0.0
            az = 0.0
            self.x = self.x * 0.70 + self.annoyed_base_x * 0.30
            self.z = self.z * 0.70 + self.annoyed_base_z * 0.30

            # Rapid vertical flinch hop around base altitude
            phase = self.annoyed_jump_ticks % 4
            if phase in (0, 1):
                self.y = self.annoyed_base_y + 0.18
                self.pitch = -12.0
            else:
                self.y = self.annoyed_base_y - 0.10
                self.pitch = 8.0
            self.vy = 0.0
            self.yaw = (self.yaw + random.uniform(-4.0, 4.0)) % 360.0

        # Soft Personal Space Repulsion (only when not feeding and not annoyed)
        dx = self.x - px
        dy = self.y - py
        dz = self.z - pz
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        h_dist = math.hypot(dx, dz)
        if food_valence <= 0.0 and not is_avoiding_hazard and not was_annoyed:
            repel_threshold = 2.0 if self.is_walking else 2.5
            if 0.01 < dist_3d < repel_threshold and abs(dy) < 2.0:
                repel_force = 0.02 * (repel_threshold - dist_3d)
                h_norm = max(0.1, h_dist)
                ax += (dx / h_norm) * repel_force
                az += (dz / h_norm) * repel_force

        # Roaming Perimeter (35m radius)
        if h_dist > 35.0:
            toward_player = math.degrees(math.atan2(-(-dx), -dz)) % 360.0
            diff = (toward_player - self.yaw + 180.0) % 360.0 - 180.0
            self.yaw_rate += math.copysign(min(2.8, abs(diff) * 0.07), diff)

        # -------------------------------------------------------------
        # 9. ALTITUDE & PITCH CONTROL (Ground Walking vs Flight)
        # -------------------------------------------------------------
        if self.is_walking:
            target_y = ground_y
            if abs(target_y - self.y) <= 1.25:
                self.y = self.y * 0.55 + target_y * 0.45
            else:
                self.y = target_y
            self.vy = 0.0

            if food_valence > 0 and food_dist < 2.5:
                self.pitch = -12.0
            elif is_escape:
                self.pitch = 5.0
            elif not was_annoyed:
                self.pitch = 0.0
        else:
            if not was_annoyed:
                self.pitch = 0.0
                if is_seeking_shelter and nearest_shelter is not None:
                    target_y = nearest_shelter[2] - 0.35
                elif is_seeking_shelter and nearest_shelter is None:
                    target_y = ground_y + 1.1
                elif is_raining and is_sheltered and nearest_shelter is not None:
                    target_y = nearest_shelter[2] - 0.45
                elif food_valence > 0 and food_dist < 3.5:
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

        # -------------------------------------------------------------
        # 10. CONTINUOUS VELOCITY INTEGRATION
        # -------------------------------------------------------------
        self.vx = self.vx * (1.0 - self.drag) + ax
        self.vz = self.vz * (1.0 - self.drag) + az

        speed = math.hypot(self.vx, self.vz)
        if self.is_walking:
            max_speed = 0.15 if (is_escape or is_avoiding_hazard) else 0.08
        else:
            max_speed = 0.26 if (is_escape or is_avoiding_hazard) else 0.12

        if speed > max_speed:
            scale = max_speed / speed
            self.vx *= scale
            self.vz *= scale

        self.x += self.vx
        self.z += self.vz

        return self.x, self.y, self.z, self.yaw, self.pitch, active_state



class WebSocketTelemetryStreamer:
    """
    Embedded High-Performance WebSocket Telemetry Server.
    Streams real-time 138k SNN spikes, dopamine, motor outputs, and Minecraft agent telemetry
    directly to the 3D WebApp Visualizer on ws://0.0.0.0:8765.
    Also receives user sensory injection commands (sweet, looming, pain, odor, grooming) from the UI.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        dev_name: str = "AMD Radeon RX 9070 XT",
        is_rocm: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.dev_name = dev_name
        self.is_rocm = is_rocm
        self.clients = set()
        self.server = None
        self.latest_packet: Optional[str] = None
        self.pending_actions: List[Dict[str, Any]] = []
        self._action_lock = threading.Lock()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    async def _handler(self, websocket) -> None:
        self.clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        print(f"[WebSocket] 3D WebApp Visualizer connected from {remote}! Active viewers: {len(self.clients)}")
        try:
            init_payload = json.dumps({
                "type": "init",
                "device": self.dev_name,
                "num_neurons": 138584,
                "num_synapses": 3732460,
                "rocm_active": self.is_rocm,
            })
            await websocket.send(init_payload)

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "action" in data:
                        with self._action_lock:
                            self.pending_actions.append(data)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[WebSocket] WebApp Visualizer disconnected. Remaining viewers: {len(self.clients)}")

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            from websockets.server import serve

        try:
            self.server = await serve(self._handler, self.host, self.port)
            print(f"[WebSocket] FlyBrain Embedded Telemetry Server listening on ws://{self.host}:{self.port}")
            print(f"[WebSocket] Open http://localhost:5173 to visualize live Minecraft fly connectome!")

            while not self._stop:
                if self.latest_packet and self.clients:
                    pkt = self.latest_packet
                    self.latest_packet = None
                    disconnected = set()
                    for c in list(self.clients):
                        try:
                            await c.send(pkt)
                        except Exception:
                            disconnected.add(c)
                    self.clients.difference_update(disconnected)
                await asyncio.sleep(0.015)
        except Exception as e:
            print(f"[WebSocket] Error starting server on port {self.port}: {e}")

    def broadcast(self, packet_dict: dict) -> None:
        self.latest_packet = json.dumps(packet_dict)

    def pop_actions(self) -> List[Dict[str, Any]]:
        with self._action_lock:
            if not self.pending_actions:
                return []
            acts = list(self.pending_actions)
            self.pending_actions.clear()
            return acts

    def stop(self) -> None:
        self._stop = True



def build_hearts_bar(health: float, max_health: float = 20.0) -> str:
    """Renders a 5-unit visual heart meter for the health bar."""
    ratio = max(0.0, min(1.0, health / max_health))
    num_full = int(round(ratio * 5))
    return "♥" * num_full + "·" * (5 - num_full)


def main():
    parser = argparse.ArgumentParser(description="FlyAI: Complete Biological Connectome Fly Agent.")
    parser.add_argument("--connectome-csv", type=str, default="data/connections_princeton.csv", help="Connections CSV")
    parser.add_argument("--annotations-csv", type=str, default="data/classification.csv", help="Annotations CSV")
    parser.add_argument("--device", type=str, default="auto", help="Target compute device ('auto', 'cuda', 'hip', 'mps', 'cpu')")
    parser.add_argument("--ticks", type=int, default=None, help="Number of ticks (default: run indefinitely)")
    parser.add_argument("--entity", type=str, default="bee", choices=["villager", "bee"], help="Entity model to embody (bee or villager)")
    parser.add_argument("--baby", action="store_true", default=False, help="Spawn as baby villager model")
    parser.add_argument("--walk", action="store_true", default=False, help="Force ground walking mode instead of aerial flight")
    parser.add_argument("--port", type=int, default=8765, help="Port to host the FlyBrain WebSocket server (default: 8765)")
    parser.add_argument("--ws-url", type=str, default=None, help="Deprecated (server is hosted on ws://0.0.0.0:<port>)")
    parser.add_argument("--no-ws", action="store_true", default=False, help="Disable WebSocket telemetry streaming")
    parser.add_argument("--rain", action="store_true", default=False, help="Simulate rain weather condition (rain avoidance & shelter seeking)")
    args = parser.parse_args()

    entity_type = args.entity
    is_baby = args.baby
    is_walking = args.walk or (entity_type == "villager" and not getattr(args, "fly", False))
    mode_str = "WALKING" if is_walking else "FLYING"

    # 1. Hardware Detection & Acceleration
    device = get_device(args.device)
    info = get_device_info(device)
    print("=" * 70)
    print(f"FlyAI: Biological Drosophila Connectome Agent [{entity_type.upper()} {mode_str}]")
    print("=" * 70)
    print(f"Processor/GPU   : {info.get('accelerator_type')} - {info.get('device_name', 'Unknown')}")
    print(f"Device Target   : {info.get('selected_device')} (PyTorch {info.get('pytorch_version')})")
    if info.get('is_rocm'):
        print(f"ROCm HIP Active : True (HIP {info.get('rocm_version')})")

    # 2. Ingest FAFB Connectome into GPU VRAM
    print(f"Loading FAFB connectome from {args.connectome_csv}...")
    t0 = time.perf_counter()
    connectome = load_connectome_from_csv(
        edges_source=args.connectome_csv,
        annotations_source=args.annotations_csv,
        device=device,
    )
    t1 = time.perf_counter()
    print(f"[OK] 138,584 neurons & 3.73M synapses loaded in {t1 - t0:.2f}s!")
    print(f"     - Nociceptive Pain Neurons : {len(connectome.nociceptive_indices):,}")
    print(f"     - Thermosensory Neurons    : {len(connectome.thermosensory_indices):,}")
    print(f"     - Ocellar Phototaxis       : {len(connectome.ocellar_indices):,}")
    print(f"     - Mechanosensory Grooming  : {len(connectome.mechanosensory_indices):,}")
    print(f"     - Circadian Sleep Neurons  : {len(connectome.circadian_indices):,}")
    print(f"     - Olfactory Odor Neurons   : {len(connectome.olfactory_indices):,}")
    print(f"     - Gustatory Taste Neurons  : {len(connectome.gustatory_indices):,}")
    print(f"     - DAN Dopaminergic Neurons : {len(connectome.dopamine_indices):,}")
    print(f"     - Kenyon Memory Neurons    : {len(connectome.kenyon_indices):,}")
    print(f"     - MBON Output Neurons      : {len(connectome.mbon_indices):,}")
    print("-" * 70)

    # 3. SNN Physics Engine
    snn = FlyBrainSNN(connectome, synaptic_gain=28.0, sensory_gain=40.0).to(device)

    # WebSocket Streamer for 3D WebApp Visualization
    streamer = None
    if not args.no_ws:
        dev_name = info.get("device_name", "AMD Radeon RX 9070 XT")
        is_rocm = info.get("is_rocm", True)
        streamer = WebSocketTelemetryStreamer(host="0.0.0.0", port=args.port, dev_name=dev_name, is_rocm=is_rocm)
        streamer.start()

    # 4. Connect to Minecraft & start background player tracker
    bridge = FastMinecraftBridge()
    bridge.initialize()
    bridge.start_background_tracker()
    time.sleep(0.5)
    px, py, pz = bridge.player_pos
    print(f"[INFO] Player detected at: X={px:.1f}, Y={py:.1f}, Z={pz:.1f}")

    flight = AerodynamicFlyAgent(px, py, pz, is_walking=is_walking)
    bridge.fly_approx_pos = (flight.x, flight.y, flight.z)

    # Fly Health State
    fly_health = 20.0
    max_health = 20.0

    def make_summon_cmd(x: float, y: float, z: float) -> str:
        if entity_type == "villager":
            model_label = "Baby Villager" if is_baby else "Villager"
            name_str = f"FAFB Connectome Fly ({model_label})"
            age_nbt = ",Age:-24000" if is_baby else ",VillagerData:{profession:'minecraft:nitwit'}"
            grav_nbt = "" if is_walking else "NoGravity:1b,"
            return (
                f"summon villager {x:.2f} {y:.2f} {z:.2f} "
                + f"{{{grav_nbt}CustomName:'\"{name_str}\"',CustomNameVisible:1b,Invulnerable:1b,PersistenceRequired:1b,Tags:[\"fly_brain_active\"]{age_nbt}}}"
            )
        else:
            name_str = "FAFB Connectome Fly (138k SNN)"
            return (
                f"summon bee {x:.2f} {y:.2f} {z:.2f} "
                + f"{{CustomName:'\"{name_str}\"',CustomNameVisible:1b,NoGravity:1b,Invulnerable:1b,PersistenceRequired:1b,Tags:[\"fly_brain_active\"]}}"
            )

    # Clean despawn any old entities, check if active entity exists, and summon exactly one
    bridge.execute([
        "tp @e[tag=fly_brain_active] ~ -300 ~",
        "tp @e[type=bee] ~ -300 ~",
        "tp @e[type=villager,tag=fly_brain_active] ~ -300 ~",
    ])
    time.sleep(0.05)
    bridge.execute([
        make_summon_cmd(flight.x, flight.y, flight.z),
        f"effect clear @e[type={entity_type},tag=fly_brain_active]",
        f'title @a actionbar [{{"text":"[FlyBrain] ","color":"gold","bold":true}},{{"text":"{entity_type.capitalize()} ({mode_str.capitalize()}) Active!","color":"green"}}]'
    ])

    print("\n[>] COMPLETE BIOLOGICAL FLY BRAIN ACTIVE!")
    print("    1. Biological Saccades (Rapid 50°-70° body turns interspersed with cruising)")
    print("    2. Positive Phototaxis (Spiraling attraction to Torches, Lanterns & Sunlight)")
    print("    3. Thermal Nociception & Avoidance (Repulsion away from Fire, Lava, Campfires)")
    print("    4. Perching & 3-Stage Stereotyped Grooming (Antennae -> Eyes -> Wings)")
    print("    5. Circadian Sleep & Quiescence (Night rest on perch, dawn awakening)")
    print("    6. Chemotaxis & Proboscis Extension (Held food & World flower attraction)")
    print("    7. Giant Fiber Looming Escape & Dopaminergic Threat Memory")
    print("    Press Ctrl+C to stop.\n", flush=True)

    tick = 0
    elapsed = 0.002

    try:
        while True:
            if args.ticks is not None and tick >= args.ticks:
                print(f"[OK] Completed {args.ticks} ticks.", flush=True)
                break

            loop_start = time.perf_counter()

            # 1. Background State
            px, py, pz = bridge.player_pos
            held_item = bridge.held_item
            is_day = bridge.is_day
            is_night = bridge.is_night
            is_raining = args.rain or bridge.is_raining
            is_sheltered = bridge.is_sheltered
            nearest_shelter = bridge.nearest_shelter
            bridge.fly_approx_pos = (flight.x, flight.y, flight.z)

            nearest_hazard = bridge.nearest_hazard
            nearest_light = bridge.nearest_light
            nearest_food_block = bridge.nearest_food_block
            nearest_perch = bridge.nearest_perch

            # 2. Evaluate Food Perception (Held Item or Nearby Block)
            held_val, held_name = FOOD_VALENCE.get(held_item, (0.0, ""))
            if held_val > 0.0:
                food_valence = held_val
                food_name = held_name
                food_dist, food_rel_angle, food_dy = flight.compute_target_relative_vector(px, py + 1.1, pz)
            elif nearest_food_block is not None and nearest_food_block[0] < 8.0:
                bdist, bx, by, bz, bval, bname = nearest_food_block
                food_valence = bval * 0.75
                food_name = bname
                food_dist, food_rel_angle, food_dy = flight.compute_target_relative_vector(bx, by + 0.5, bz)
            else:
                food_valence = 0.0
                food_name = ""
                food_dist, food_rel_angle, food_dy = 10.0, 0.0, 0.0

            is_feeding = (food_valence > 0.0 and food_dist < 2.2)

            # 3. Check for Looming Threat (respecting 60° rear blind spot) & Johnston's Organ
            is_threat, threat_rel_angle = flight.detect_looming_threat(px, py, pz)
            is_sound_touch = flight.detect_sound_or_touch(px, py, pz)

            if (is_threat or is_sound_touch or (food_valence > 0.0 and food_dist < 2.2)) and flight.is_sleeping:
                flight.is_sleeping = False
                flight.is_landed = flight.is_walking
                snn.set_circadian_phase(is_night=False)

            # Circadian sleep-wake modulation
            snn.set_circadian_phase(is_night=is_night)

            # Update hunger state in SNN (<50% flight energy activates hunger neurons)
            if flight.flight_energy < 50.0:
                snn.hunger_level = max(0.0, min(1.0, (50.0 - flight.flight_energy) / 50.0))
            else:
                snn.hunger_level = 0.0

            # 4. Sensory input encoding into 138k SNN
            if flight.is_sleeping or (is_night and not is_escape and flight.annoyed_jump_ticks <= 0):
                # Biological Sensory Gating during nocturnal sleep:
                ext_current = torch.zeros(snn.num_neurons, device=device)
                circ_idx = getattr(snn.connectome, "circadian_indices", None)
                if circ_idx is not None and len(circ_idx) > 0:
                    ext_current[circ_idx] = 12.0
                spontaneous = (torch.rand(snn.num_neurons, device=device) < 0.00012).float() * 8.0
                ext_current = ext_current + spontaneous
            else:
                eye_left, eye_right = flight.compute_sensory_inputs(px, py, pz, tick)
                ext_current = snn.encode_sensory_input(eye_left, eye_right)
                spontaneous = (torch.rand(snn.num_neurons, device=device) < 0.0035).float() * 14.0
                ext_current = ext_current + spontaneous

            # Johnston's Organ Chordotonal Transduction (Sound as Physical Touch within 1.5m)
            if is_sound_touch or flight.annoyed_jump_ticks > 0:
                mech_sound_cur = snn.inject_johnstons_organ_sound_touch(intensity=1.0)
                ext_current = ext_current + mech_sound_cur

            # Chemosensory Odor Ingestion
            if food_valence > 0 and not flight.is_sleeping:
                odor_conc = food_valence / (1.0 + 0.15 * max(0.0, food_dist - 1.0))
                odor_current = snn.inject_food_odor(
                    valence=odor_conc,
                    rel_angle=food_rel_angle,
                    is_feeding=is_feeding,
                )
                ext_current = ext_current + odor_current

            # Phototaxis Ocellar Light Ingestion
            if nearest_light is not None and nearest_light[0] < 10.0 and not flight.is_sleeping and not is_raining:
                ldist, lx, ly, lz, _ = nearest_light
                _, l_rel_angle, _ = flight.compute_target_relative_vector(lx, ly, lz)
                l_intensity = max(0.0, min(1.0, (10.0 - ldist) / 10.0))
                light_current = snn.inject_light_stimulus(intensity=l_intensity, rel_angle=l_rel_angle)
                ext_current = ext_current + light_current

            # Thermonociceptive Hazard Ingestion
            if nearest_hazard is not None and nearest_hazard[0] < 5.0:
                hdist, hx, hy, hz, _ = nearest_hazard
                if hdist < 2.0 and flight.is_sleeping:
                    flight.is_sleeping = False
                    flight.is_landed = flight.is_walking
                _, h_rel_angle, _ = flight.compute_target_relative_vector(hx, hy, hz)
                h_intensity = max(0.0, min(1.0, (5.0 - hdist) / 5.0))
                hazard_current = snn.inject_thermal_hazard(intensity=h_intensity, rel_angle=h_rel_angle)
                ext_current = ext_current + hazard_current
                if hdist < 1.6 and tick % 10 == 0:
                    fly_health = max(1.0, fly_health - 2.0)  # Heat damage

            # Grooming Mechanosensory Ingestion
            if flight.is_grooming:
                grooming_current = snn.inject_mechanosensory_grooming()
                ext_current = ext_current + grooming_current

            # Threat Nociception & Giant Fiber Reflex (only if not in rear blind spot)
            if is_threat:
                pain_current = snn.inject_pain(intensity=1.0, threat_rel_angle=threat_rel_angle, world_heading=flight.yaw)
                ext_current = ext_current + pain_current

            # Rain Nociception & Conditioned Shelter Seeking
            if is_raining:
                if not is_sheltered:
                    # Calculate relative angle towards nearest shelter (if known)
                    shelter_rel = None
                    if nearest_shelter is not None:
                        _, sx, sy, sz = nearest_shelter
                        _, shelter_rel, _ = flight.compute_target_relative_vector(sx, sy, sz)

                    # Nociceptive rain droplet bombardment + mushroom body threat conditioning
                    rain_pain_cur = snn.inject_rain_pain(
                        intensity=1.0,
                        shelter_rel_angle=shelter_rel,
                        world_heading=flight.yaw,
                    )
                    ext_current = ext_current + rain_pain_cur

                    # Mechanical rain damage when exposed (droplet impact trauma)
                    if tick % 30 == 0:
                        fly_health = max(1.0, fly_health - 1.0)
                else:
                    # Safe under cover: relief reward dopamine (PAM reinforcement)
                    shelter_relief_cur = snn.inject_shelter_relief()
                    ext_current = ext_current + shelter_relief_cur
                    if tick % 30 == 0:
                        fly_health = min(max_health, fly_health + 0.5)

            # Feeding Nutrient Regeneration
            if is_feeding and tick % 15 == 0:
                fly_health = min(max_health, fly_health + 1.0)

            # Process real-time sensory injection requests from 3D WebApp Visualizer
            if streamer is not None:
                for act in streamer.pop_actions():
                    action_name = act.get("action")
                    intensity = float(act.get("intensity", 1.0))
                    if action_name in ("inject_sweet", "inject_looming", "inject_pain", "inject_rain_pain", "inject_odor", "inject_sound_touch"):
                        flight.is_sleeping = False
                        flight.is_landed = flight.is_walking
                        snn.set_circadian_phase(is_night=False)
                    if action_name == "inject_sweet":
                        sweet_current = snn.inject_food_odor(valence=intensity, is_feeding=True)
                        ext_current = ext_current + sweet_current
                        fly_health = min(max_health, fly_health + 2.0)
                    elif action_name == "inject_looming":
                        threat_current = snn.inject_pain(intensity=intensity, threat_rel_angle=0.0, world_heading=flight.yaw)
                        ext_current = ext_current + threat_current
                    elif action_name == "inject_pain":
                        pain_cur = snn.inject_pain(intensity=intensity, threat_rel_angle=0.0, world_heading=flight.yaw)
                        ext_current = ext_current + pain_cur
                    elif action_name == "inject_odor":
                        odor_cur = snn.inject_food_odor(valence=intensity, is_feeding=False)
                        ext_current = ext_current + odor_cur
                    elif action_name == "inject_sound_touch":
                        flight.annoyed_jump_ticks = 16
                        flight.annoyed_base_x = flight.x
                        flight.annoyed_base_y = flight.y
                        flight.annoyed_base_z = flight.z
                        flight.annoyed_cooldown = 40
                        ext_current = ext_current + snn.inject_johnstons_organ_sound_touch(intensity=intensity)
                    elif action_name == "inject_grooming":
                        flight.is_grooming = True
                        flight.grooming_tick = 0
                    elif action_name in ("set_rain", "toggle_rain"):
                        bridge.is_raining = not bridge.is_raining if action_name == "toggle_rain" else bool(act.get("is_raining", True))
                    elif action_name == "reset":
                        snn.reset_state()
                    elif action_name == "set_phase":
                        is_night = bool(act.get("is_night", False))
                        snn.set_circadian_phase(is_night=is_night)

            # 5. Advance 138k SNN step on GPU (~2.1ms latency)
            spikes = snn.step(external_current=ext_current)

            # 6. Decode descending motor output & escape state
            turn, forward, total_spikes, is_escape = snn.decode_motor_output(spikes, return_escape=True)

            # 7. Query Mushroom Body learned threat avoidance
            learned_steer = snn.get_learned_avoidance_steer(flight.yaw)

            # 8. Advance locomotion physics (walking/flight) & biological state machine
            ground_y = bridge.get_ground_y(flight.x, flight.z, py)
            bx, by, bz, byaw, bpitch, current_state_label = flight.step_behavior_and_physics(
                turn_signal=turn,
                forward_signal=forward,
                learned_steer=learned_steer,
                px=px,
                py=py,
                pz=pz,
                tick=tick,
                is_escape=is_escape,
                food_valence=food_valence,
                food_dist=food_dist,
                food_rel_angle=food_rel_angle,
                food_dy=food_dy,
                food_name=food_name,
                nearest_hazard=nearest_hazard,
                nearest_light=nearest_light,
                nearest_perch=nearest_perch,
                nearest_shelter=nearest_shelter,
                is_night=is_night,
                is_raining=is_raining,
                is_sheltered=is_sheltered,
                ground_y=ground_y,
            )

            # Enrich state label if feeding or hungry
            if current_state_label == "FEEDING":
                current_state_label = f"FEEDING ({food_name or 'Sugar'}) [+HP]"
            elif current_state_label == "HUNGRY":
                current_state_label = f"HUNGRY ({food_name or 'Melon/Sugar'})"

            # 9. Send Minecraft Teleport packet with exact tag selector and pitch
            tp_cmd = f"tp @e[type={entity_type},tag=fly_brain_active,limit=1] {bx:.2f} {by:.2f} {bz:.2f} {byaw:.1f} {bpitch:.1f}"
            cmds = [tp_cmd]

            # Annoyance smoke particles when disturbed by sound/touch within 1.5 blocks:
            if flight.annoyed_jump_ticks in (16, 15, 14, 13):
                cmds.append(f"particle smoke {bx:.2f} {by + 0.2:.2f} {bz:.2f} 0.15 0.15 0.15 0.02 6 force")

            # 10. Update Two-Line In-Game Actionbar HUD every 2 ticks (100ms)
            if tick % 2 == 0:
                hearts_bar = build_hearts_bar(fly_health, max_health)
                pam_val = snn.appetitive_dopamine

                if "ESCAPE" in current_state_label:
                    state_col = "dark_red"
                elif "AVOIDANCE" in current_state_label:
                    state_col = "red"
                elif "ANNOYED" in current_state_label:
                    state_col = "dark_purple"
                elif "FEEDING" in current_state_label:
                    state_col = "light_purple"
                elif "HUNGRY" in current_state_label:
                    state_col = "gold"
                elif "SEEKING SHELTER" in current_state_label:
                    state_col = "red"
                elif "SHELTERED" in current_state_label:
                    state_col = "blue"
                elif "PHOTOTAXIS" in current_state_label:
                    state_col = "yellow"
                elif "GROOMING" in current_state_label or "PERCHING" in current_state_label:
                    state_col = "aqua"
                elif "SLEEPING" in current_state_label:
                    state_col = "dark_aqua"
                elif "Saccade" in current_state_label:
                    state_col = "dark_aqua"
                else:
                    state_col = "green"

                # Alternates between Line 1 and Line 2 every 30 ticks (1.5s) to eliminate right-side text clipping
                hud_phase = (tick // 30) % 2
                if hud_phase == 0:
                    # Line 1: Identity, HP, State (~45 chars)
                    hud_cmd = (
                        f'title @a actionbar ['
                        f'{{"text":"[FAFB {entity_type.capitalize()}] ","color":"gold","bold":true}},'
                        f'{{"text":"HP: {int(fly_health)} [{hearts_bar}] ","color":"red","bold":true}},'
                        f'{{"text":"[{current_state_label}]","color":"{state_col}","bold":true}}]'
                    )
                else:
                    # Line 2: SNN Spikes, Energy %, PAM-DA, Rear Blind Spot, Overhead Cover (~50 chars)
                    blind_tag = "BLIND" if flight.in_rear_blind_spot else "VIS"
                    blind_col = "green" if flight.in_rear_blind_spot else "gray"
                    cov_tag = "Sheltered" if is_sheltered else "Open"
                    cov_col = "aqua" if is_sheltered else ("gold" if is_raining else "white")

                    hud_cmd = (
                        f'title @a actionbar ['
                        f'{{"text":"Spikes: {total_spikes:,} ","color":"aqua"}},'
                        f'{{"text":"| Nrg: {int(flight.flight_energy)}% ","color":"gold"}},'
                        f'{{"text":"| PAM: {pam_val:.2f} ","color":"yellow"}},'
                        f'{{"text":"| Rear: [{blind_tag}] ","color":"{blind_col}"}},'
                        f'{{"text":"| Cov: {cov_tag}","color":"{cov_col}"}}]'
                    )
                cmds.append(hud_cmd)

            res = bridge.execute(cmds)
            res_str = json.dumps(res)

            # Broadcast real-time spikes & Minecraft agent telemetry to WebApp
            if streamer is not None:
                firing_indices = torch.nonzero(spikes).flatten().tolist()
                speed_mps = math.hypot(flight.vx, flight.vz) * 20.0
                dx_p = px - flight.x
                dz_p = pz - flight.z
                p_dist = math.hypot(dx_p, dz_p)

                streamer.broadcast({
                    "type": "telemetry",
                    "step": tick,
                    "spikes": firing_indices,
                    "count": len(firing_indices),
                    "dopamine": round(float(getattr(snn, "dopamine_level", 0.0)), 4),
                    "appetitive": round(float(getattr(snn, "appetitive_dopamine", 0.0)), 4),
                    "escape_timer": getattr(snn, "escape_timer", 0),
                    "is_escaping": bool(is_escape),
                    "is_sleeping": bool(flight.is_sleeping),
                    "turn": round(float(turn), 3),
                    "forward": round(float(forward), 3),
                    "latency_ms": round(elapsed * 1000.0, 2),
                    "agent": {
                        "connected": True,
                        "source": "Minecraft",
                        "entity": entity_type,
                        "pos": [round(bx, 2), round(by, 2), round(bz, 2)],
                        "yaw": round(byaw, 1),
                        "pitch": round(bpitch, 1),
                        "hp": round(fly_health, 1),
                        "max_hp": round(max_health, 1),
                        "energy": round(flight.flight_energy, 1),
                        "in_blind_spot": bool(flight.in_rear_blind_spot),
                        "is_raining": bool(is_raining),
                        "is_sheltered": bool(is_sheltered),
                        "hunger": round(float(snn.hunger_level), 2),
                        "state": current_state_label,
                        "speed": round(speed_mps, 2),
                        "nearest_light": {"dist": round(nearest_light[0], 1), "name": nearest_light[4]} if nearest_light else None,
                        "nearest_hazard": {"dist": round(nearest_hazard[0], 1), "name": nearest_hazard[4]} if nearest_hazard else None,
                        "nearest_food": {"dist": round(nearest_food_block[0], 1), "name": nearest_food_block[5]} if nearest_food_block else None,
                        "player_dist": round(p_dist, 1),
                        "is_night": is_night,
                        "last_seen": round(time.time(), 2),
                    }
                })

            # Auto-respawn fallback if entity was destroyed by player or game
            if "No entity was found" in res_str:
                flight.x, flight.y, flight.z = px + 1.5, py + 1.8, pz + 1.5
                flight.is_landed = False
                flight.is_grooming = False
                flight.is_sleeping = False
                bridge.execute([
                    make_summon_cmd(flight.x, flight.y, flight.z),
                    f"effect clear @e[type={entity_type},tag=fly_brain_active]"
                ])

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.002, 0.05 - elapsed)
            time.sleep(sleep_time)

            if tick % 20 == 0:
                speed_mps = math.hypot(flight.vx, flight.vz) * 20.0
                print(
                    f"[Tick {tick:4d}] Spikes: {total_spikes:6,d} | State: {current_state_label[:26]:26s} | "
                    f"HP: {fly_health:4.1f} | Turn: {turn:+5.1f}° | Speed: {speed_mps:4.1f}m/s | "
                    f"Energy: {flight.flight_energy:4.0f}% | Pos: ({bx:.1f}, {by:.1f}, {bz:.1f})",
                    flush=True,
                )

            tick += 1

    except KeyboardInterrupt:
        print("\n[INFO] Autonomous flight stopped by user.", flush=True)
    finally:
        if streamer is not None:
            streamer.stop()
        bridge.stop_background_tracker()


if __name__ == "__main__":
    main()
