"""
Main Application Entrypoint for Drosophila Connectome SNN Telemetry Engine.

Loads connectome data (either from FAFB CSV or synthetic circuit generator),
initializes the FlyBrainSNN model on the selected hardware target (ROCm/CUDA/CPU),
and runs the real-time WebSocket telemetry bridge.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from connectome_loader import generate_synthetic_connectome, load_connectome_from_csv
from device_utils import get_device, get_device_info
from snn_engine import FlyBrainSNN
from websocket_server import FlyTelemetryServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flyai.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlyAI: Connectome-Driven Spiking Neural Network (SNN) Virtual Agent Engine"
    )
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket bind host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket bind port (default: 8765)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'hip', 'mps', or 'cpu')")
    parser.add_argument("--connectome-csv", type=str, default=None, help="Path to FAFB edges CSV file")
    parser.add_argument("--annotations-csv", type=str, default=None, help="Path to neuron annotations CSV file")
    parser.add_argument("--num-visual", type=int, default=64, help="Visual input neurons for synthetic circuit")
    parser.add_argument("--num-steering", type=int, default=128, help="Steering neurons for synthetic circuit")
    parser.add_argument("--num-motor", type=int, default=32, help="Motor neurons for synthetic circuit")
    parser.add_argument("--density", type=float, default=0.05, help="Synaptic density for synthetic circuit")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    # 1. Hardware device configuration
    device = get_device(args.device)
    dev_info = get_device_info(device)
    logger.info("=" * 60)
    logger.info("Initializing FlyAI Connectome SNN Engine")
    logger.info("=" * 60)
    logger.info(f"Resolved Device : {dev_info['selected_device']}")
    logger.info(f"CUDA Available  : {dev_info['cuda_available']}")
    logger.info(f"AMD ROCm Active : {dev_info['is_rocm']} (HIP: {dev_info.get('rocm_version')})")
    if "device_name" in dev_info:
        logger.info(f"GPU Hardware    : {dev_info['device_name']}")
    logger.info(f"PyTorch Version : {dev_info['pytorch_version']}")
    logger.info("-" * 60)

    # 2. Connectome Ingestion
    if args.connectome_csv:
        logger.info(f"Loading FAFB connectome from: {args.connectome_csv}")
        connectome = load_connectome_from_csv(
            edges_source=args.connectome_csv,
            annotations_source=args.annotations_csv,
            device=device,
        )
    else:
        logger.info(
            f"Synthesizing Drosophila connectome: Visual={args.num_visual}, "
            f"Steering={args.num_steering}, Motor={args.num_motor}, Density={args.density}"
        )
        connectome = generate_synthetic_connectome(
            num_visual=args.num_visual,
            num_steering=args.num_steering,
            num_motor=args.num_motor,
            density=args.density,
            device=device,
        )

    logger.info(f"Total Neurons   : {connectome.num_neurons}")
    logger.info(f"Visual Neurons  : {len(connectome.visual_indices)} (Left: {len(connectome.visual_left_indices)}, Right: {len(connectome.visual_right_indices)})")
    logger.info(f"Steering Neurons: {len(connectome.steering_indices)}")
    logger.info(f"Motor Neurons   : {len(connectome.motor_indices)} (Left: {len(connectome.motor_left_indices)}, Right: {len(connectome.motor_right_indices)})")
    logger.info(f"Synaptic Edges  : {connectome.weight_matrix._nnz()}")
    logger.info("-" * 60)

    # 3. SNN Simulator
    snn = FlyBrainSNN(connectome=connectome)
    logger.info("FlyBrainSNN initialized with LIF dynamics.")

    # 4. WebSocket Server
    server = FlyTelemetryServer(snn=snn, host=args.host, port=args.port)

    # Handle graceful exit
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Termination signal received. Shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except (NotImplementedError, AttributeError):
            # Windows signal handler fallback
            pass

    await server.start()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
        logger.info("Shutdown complete.")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Exiting on KeyboardInterrupt.")
        sys.exit(0)


if __name__ == "__main__":
    main()
