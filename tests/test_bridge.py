"""
Tests for Real-Time Telemetry WebSocket Bridge.
"""

import asyncio
import json
import socket
import pytest
import websockets
try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect

from connectome_loader import generate_synthetic_connectome
from snn_engine import FlyBrainSNN
from websocket_server import FlyTelemetryServer


def get_free_port() -> int:
    """Find a currently available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def test_snn():
    """Fixture providing a lightweight SNN instance."""
    connectome = generate_synthetic_connectome(
        num_visual=16,
        num_steering=32,
        num_motor=8,
        density=0.1,
        seed=42,
        device="cpu",
    )
    return FlyBrainSNN(connectome)


@pytest.mark.asyncio
async def test_process_payload_direct(test_snn):
    """Verify payload processing directly without socket overhead."""
    server = FlyTelemetryServer(snn=test_snn, host="127.0.0.1", port=8765)
    payload = json.dumps({"sensors": [0.8, 0.2]})

    response = await server.process_payload(payload)

    assert "turn" in response
    assert "forward" in response
    assert "total_spikes" in response
    assert isinstance(response["turn"], float)
    assert isinstance(response["forward"], float)
    assert isinstance(response["total_spikes"], int)
    assert response["forward"] >= 0.0
    assert response["total_spikes"] >= 0


@pytest.mark.asyncio
async def test_invalid_payload_error(test_snn):
    """Verify server raises descriptive error on malformed payload."""
    server = FlyTelemetryServer(snn=test_snn, host="127.0.0.1", port=8765)

    with pytest.raises(ValueError):
        await server.process_payload(json.dumps({"sensors": [1.0]}))  # Only 1 element

    with pytest.raises(ValueError):
        await server.process_payload(json.dumps({"wrong_key": [1.0, 2.0]}))


@pytest.mark.asyncio
async def test_websocket_client_communication(test_snn):
    """
    Test live client-server interaction over an actual WebSocket connection.
    Verifies JSON protocol, motor responses, and data integrity.
    """
    port = get_free_port()
    server = FlyTelemetryServer(snn=test_snn, host="127.0.0.1", port=port)
    await server.start()

    uri = f"ws://127.0.0.1:{port}"
    try:
        async with connect(uri) as client:
            # Send valid sensory raycast input
            req = {"sensors": [0.75, 0.25]}
            await client.send(json.dumps(req))

            raw_resp = await asyncio.wait_for(client.recv(), timeout=3.0)
            data = json.loads(raw_resp)

            assert "turn" in data, "Response missing 'turn' key."
            assert "forward" in data, "Response missing 'forward' key."
            assert "total_spikes" in data, "Response missing 'total_spikes' key."

            assert isinstance(data["turn"], (int, float))
            assert isinstance(data["forward"], (int, float))
            assert isinstance(data["total_spikes"], int)
            assert data["forward"] >= 0.0

            # Send second payload to verify continuous stepping
            req2 = {"sensors": [0.1, 0.9]}
            await client.send(json.dumps(req2))
            raw_resp2 = await asyncio.wait_for(client.recv(), timeout=3.0)
            data2 = json.loads(raw_resp2)

            assert "turn" in data2
            assert "forward" in data2
            assert "total_spikes" in data2

    finally:
        await server.stop()
