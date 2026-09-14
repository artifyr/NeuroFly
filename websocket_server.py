"""
Real-Time Telemetry WebSocket Bridge for Drosophila SNN Agent.

Listens on ws://localhost:8765 (configurable), receives raycast sensory data,
drives the FlyBrainSNN simulation step, and returns motor telemetry (turn, forward, total_spikes).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Union

import torch
import websockets
try:
    from websockets.asyncio.server import Server, ServerConnection, serve
except ImportError:  # Fallback for older websockets versions
    from websockets.server import WebSocketServer as Server, WebSocketServerProtocol as ServerConnection, serve

from snn_engine import FlyBrainSNN

logger = logging.getLogger(__name__)


class FlyTelemetryServer:
    """
    WebSocket server that interfaces an external 3D virtual environment
    with the connectome-driven FlyBrainSNN.
    """

    def __init__(
        self,
        snn: FlyBrainSNN,
        host: str = "localhost",
        port: int = 8765,
    ) -> None:
        self.snn = snn
        self.host = host
        self.port = port
        self.server: Optional[Server] = None
        self._stop_event = asyncio.Event()

    async def process_payload(self, raw_message: Union[str, bytes]) -> dict:
        """
        Process a sensory payload and advance the SNN by one timestep.

        Expected JSON format:
            {"sensors": [raycast_distance_left, raycast_distance_right]}

        Returns dict:
            {"turn": float, "forward": float, "total_spikes": int}
        """
        data = json.loads(raw_message)
        sensors = data.get("sensors")

        if sensors is None or not isinstance(sensors, (list, tuple)) or len(sensors) < 2:
            raise ValueError("Payload must contain 'sensors' list with at least 2 float elements.")

        ray_left = float(sensors[0])
        ray_right = float(sensors[1])

        # 1. Encode sensory reading into visual input currents
        ext_current = self.snn.encode_sensory_input(ray_left, ray_right)

        # 2. Add realistic spontaneous biological background noise (0.3% Poisson baseline)
        # Keeps recurrent Central Complex circuits perpetually active like a living brain
        spontaneous = (torch.rand(self.snn.num_neurons, device=self.snn.device) < 0.003).float() * 12.0
        ext_current = ext_current + spontaneous

        # 3. Advance simulation by 1 timestep
        spikes = self.snn.step(external_current=ext_current)

        # 4. Decode motor commands from descending motor neurons
        turn_val, forward_val, total_spikes = self.snn.decode_motor_output(spikes)

        return {
            "turn": float(turn_val),
            "forward": float(forward_val),
            "total_spikes": int(total_spikes),
        }

    async def client_handler(self, websocket: ServerConnection, *args, **kwargs) -> None:
        """Handle individual WebSocket client connection session."""
        remote_address = getattr(websocket, "remote_address", "unknown")
        logger.info(f"Client connected: {remote_address}")

        try:
            async for message in websocket:
                try:
                    response_dict = await self.process_payload(message)
                    await websocket.send(json.dumps(response_dict))
                except json.JSONDecodeError:
                    error_resp = {"error": "Invalid JSON format."}
                    await websocket.send(json.dumps(error_resp))
                except Exception as err:
                    logger.error(f"Error handling message: {err}")
                    error_resp = {"error": str(err)}
                    await websocket.send(json.dumps(error_resp))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {remote_address}")
        finally:
            logger.info(f"Connection closed for: {remote_address}")

    async def start(self) -> Server:
        """Start listening for incoming WebSocket connections."""
        self.server = await serve(self.client_handler, self.host, self.port)
        logger.info(f"Fly SNN Telemetry Server listening on ws://{self.host}:{self.port}")
        return self.server

    async def stop(self) -> None:
        """Gracefully close the server and terminate active client connections."""
        self._stop_event.set()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            logger.info("Fly SNN Telemetry Server stopped.")

    async def run(self) -> None:
        """Run server continuously until stopped or cancelled."""
        await self.start()
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()
