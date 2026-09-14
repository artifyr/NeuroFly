import pytest
import torch
from connectome_loader import generate_synthetic_connectome
from snn_engine import FlyBrainSNN

def test_thermosensory_hazard():
    connectome = generate_synthetic_connectome(num_visual=32, num_steering=64, num_motor=16, seed=42)
    snn = FlyBrainSNN(connectome)

    # Inject thermal hazard on the left
    cur = snn.inject_thermal_hazard(intensity=1.0, rel_angle=-45.0)
    assert cur.shape == (snn.num_neurons,)
    assert cur.sum() > 0.0

    # Thermosensory and motor neurons should be activated
    assert snn.dopamine_level > 0.0
    spikes = snn.step(cur)
    turn, fwd, total = snn.decode_motor_output(spikes)
    # Heat on left should drive right turn (positive turn signal)
    assert turn > 0.0

def test_phototaxis_light_stimulus():
    connectome = generate_synthetic_connectome(num_visual=32, num_steering=64, num_motor=16, seed=42)
    snn = FlyBrainSNN(connectome)

    # Light on the right
    cur = snn.inject_light_stimulus(intensity=1.0, rel_angle=45.0)
    assert cur.shape == (snn.num_neurons,)
    assert cur.sum() > 0.0

    spikes = snn.step(cur)
    turn, fwd, total = snn.decode_motor_output(spikes)
    # Light on right should bias right turn
    assert turn > 0.0

def test_mechanosensory_grooming():
    connectome = generate_synthetic_connectome(num_visual=32, num_steering=64, num_motor=16, seed=42)
    snn = FlyBrainSNN(connectome)

    cur = snn.inject_mechanosensory_grooming()
    assert snn.is_grooming is True
    assert cur.shape == (snn.num_neurons,)

def test_circadian_phase_sleep():
    connectome = generate_synthetic_connectome(num_visual=32, num_steering=64, num_motor=16, seed=42)
    snn = FlyBrainSNN(connectome)

    # Day time: not sleeping
    snn.set_circadian_phase(is_night=False)
    assert snn.is_sleeping is False

    # Night time: quiescent sleep
    snn.set_circadian_phase(is_night=True)
    assert snn.is_sleeping is True
