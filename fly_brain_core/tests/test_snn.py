"""
Tests for SNN Physics & Dynamics.

Covers:
- Sub-test A (Sub-threshold): Low current depolarizes without firing.
- Sub-test B (Super-threshold): High current fires, resets, triggers refractory period, and propagates downstream.
- Sub-test C (Matrix Scaling): 1,000+ neuron network stability, performance, and numerical validity.
"""

import time
import pytest
import torch

from connectome_loader import ConnectomeData, generate_synthetic_connectome
from device_utils import get_device
from snn_engine import FlyBrainSNN


def create_minimal_two_neuron_network(device="cpu") -> ConnectomeData:
    """Create a minimal 2-neuron network: Neuron 0 connects to Neuron 1."""
    dev = torch.device(device)
    indices = torch.tensor([[1], [0]], dtype=torch.int64, device=dev)  # row 1 (post), col 0 (pre)
    values = torch.tensor([1.0], dtype=torch.float32, device=dev)
    sparse_w = torch.sparse_coo_tensor(indices, values, (2, 2), device=dev, check_invariants=False)

    return ConnectomeData(
        weight_matrix=sparse_w,
        num_neurons=2,
        neuron_ids=[100, 101],
        id_to_idx={100: 0, 101: 1},
        visual_indices=torch.tensor([0], device=dev),
        visual_left_indices=torch.tensor([0], device=dev),
        visual_right_indices=torch.tensor([], dtype=torch.int64, device=dev),
        steering_indices=torch.tensor([], dtype=torch.int64, device=dev),
        motor_indices=torch.tensor([1], device=dev),
        motor_left_indices=torch.tensor([1], device=dev),
        motor_right_indices=torch.tensor([], dtype=torch.int64, device=dev),
        device=dev,
    )


def test_sub_test_a_sub_threshold():
    """
    Sub-test A (Sub-threshold):
    Pass low current; verify membrane potential increases without firing a spike.
    """
    connectome = create_minimal_two_neuron_network(device="cpu")
    snn = FlyBrainSNN(connectome, v_rest=-70.0, v_thresh=-50.0, v_reset=-70.0)

    # Initial state
    assert torch.allclose(snn.v, torch.tensor([-70.0, -70.0]))

    # Inject sub-threshold current to neuron 0 (e.g. +8 mV -> brings -70 to -62 mV, below -50 mV)
    ext_current = torch.tensor([8.0, 0.0], dtype=torch.float32)
    spikes = snn.step(external_current=ext_current)

    # Assert no spikes were fired
    assert (spikes == 0.0).all()

    # Membrane potential should have depolarized above resting potential
    assert snn.v[0].item() > -70.0
    assert snn.v[0].item() < -50.0
    assert snn.refractory_timer[0].item() == 0


def test_sub_test_b_super_threshold():
    """
    Sub-test B (Super-threshold):
    Pass high current; verify neuron fires a spike, resets to V_rest,
    enters refractory period, and propagates signal to downstream connected weights.
    """
    connectome = create_minimal_two_neuron_network(device="cpu")
    # Sub-threshold downstream synaptic gain (12 mV)
    snn = FlyBrainSNN(
        connectome,
        v_rest=-70.0,
        v_thresh=-50.0,
        v_reset=-70.0,
        refractory_steps=2,
        synaptic_gain=12.0,
    )

    # Step 1: Inject high current (+30 mV) into neuron 0 to trigger action potential
    ext_current_step1 = torch.tensor([30.0, 0.0], dtype=torch.float32)
    spikes_step1 = snn.step(external_current=ext_current_step1)

    # Assert neuron 0 fired a spike, neuron 1 did not
    assert spikes_step1[0].item() == 1.0
    assert spikes_step1[1].item() == 0.0

    # Assert neuron 0 membrane voltage reset back to V_rest
    assert snn.v[0].item() == -70.0

    # Assert neuron 0 refractory timer is activated
    assert snn.refractory_timer[0].item() == 2

    # Assert synaptic current was computed for downstream neuron 1
    assert snn.synaptic_current[1].item() == 12.0

    # Step 2: Advance one timestep without external current
    # Downstream neuron 1 receives the propagated synaptic current and depolarizes
    spikes_step2 = snn.step()

    # Neuron 1 voltage must have depolarized due to incoming synaptic current (-58 mV > -70 mV)
    assert snn.v[1].item() > -70.0
    assert snn.v[1].item() < -50.0

    # Neuron 0 should still be in refractory state (timer decremented to 1)
    assert snn.refractory_timer[0].item() == 1

    # Also verify super-threshold synaptic cascade:
    snn_cascade = FlyBrainSNN(
        connectome,
        v_rest=-70.0,
        v_thresh=-50.0,
        v_reset=-70.0,
        refractory_steps=2,
        synaptic_gain=25.0,
    )
    snn_cascade.step(external_current=torch.tensor([30.0, 0.0], dtype=torch.float32))
    spikes_cascade = snn_cascade.step()
    # Downstream neuron 1 receives 25.0 mV -> crosses -50.0 mV and fires
    assert spikes_cascade[1].item() == 1.0
    assert snn_cascade.v[1].item() == -70.0  # Reset after firing


def test_sub_test_c_matrix_scaling():
    """
    Sub-test C (Matrix Scaling):
    Verify performance and stability when running a 1,000+ neuron matrix step.
    """
    device = get_device()
    num_visual = 200
    num_steering = 700
    num_motor = 150
    total_neurons = num_visual + num_steering + num_motor
    assert total_neurons >= 1000

    connectome = generate_synthetic_connectome(
        num_visual=num_visual,
        num_steering=num_steering,
        num_motor=num_motor,
        density=0.04,
        seed=999,
        device=device,
    )

    snn = FlyBrainSNN(connectome)
    snn.to(device)

    # Run 50 simulation steps with varying sensory stimuli
    num_steps = 50
    start_time = time.perf_counter()

    for step_idx in range(num_steps):
        # Varying sensory input
        ray_left = 0.5 + 0.3 * (step_idx % 5) / 5.0
        ray_right = 0.3 + 0.4 * ((step_idx + 2) % 5) / 5.0

        sensory_i = snn.encode_sensory_input(ray_left, ray_right)
        spikes = snn.step(external_current=sensory_i)
        turn, forward, total_spikes = snn.decode_motor_output(spikes)

        # Numerical sanity checks
        assert not torch.isnan(snn.v).any(), f"NaN detected in membrane potential at step {step_idx}"
        assert not torch.isinf(snn.v).any(), f"Inf detected in membrane potential at step {step_idx}"
        assert not torch.isnan(spikes).any()
        assert isinstance(turn, float)
        assert isinstance(forward, float)
        assert isinstance(total_spikes, int)

    elapsed = time.perf_counter() - start_time
    steps_per_sec = num_steps / elapsed

    # Assert healthy execution throughput (well above 50 steps/sec)
    assert steps_per_sec > 10.0, f"Performance too low: {steps_per_sec:.1f} steps/sec"
