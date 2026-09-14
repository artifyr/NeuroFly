"""
Tests for Biological Nociceptive Pain, Escape Reflexes, and Dopaminergic Threat Learning.
"""

import pytest
import torch

from connectome_loader import generate_synthetic_connectome
from snn_engine import FlyBrainSNN


@pytest.fixture
def test_snn():
    conn = generate_synthetic_connectome(num_visual=32, num_steering=64, num_motor=16, device="cpu")
    return FlyBrainSNN(conn, synaptic_gain=25.0, sensory_gain=30.0)


def test_pain_injection_and_dopamine(test_snn):
    """Verify pain injection depolarizes nociceptors and raises dopamine level."""
    assert test_snn.dopamine_level == 0.0
    assert not test_snn.is_escaping

    pain_current = test_snn.inject_pain(intensity=1.0, threat_rel_angle=-45.0, world_heading=90.0)

    assert pain_current.shape == (test_snn.num_neurons,)
    assert pain_current.sum() > 0.0
    assert test_snn.dopamine_level > 0.5
    assert test_snn.is_escaping

    # Step forward and verify escape timer counts down and dopamine decays
    spikes = test_snn.step(pain_current)
    assert spikes.sum() > 0
    assert test_snn.dopamine_level < 1.0


def test_giant_fiber_contralateral_escape(test_snn):
    """
    Threat on LEFT (rel_angle < 0) should inject into RIGHT motor neurons,
    driving a positive turn (banking right).
    """
    pain_current = test_snn.inject_pain(intensity=1.0, threat_rel_angle=-60.0, world_heading=0.0)
    spikes = test_snn.step(pain_current)
    turn, fwd, total, is_escape = test_snn.decode_motor_output(spikes, return_escape=True)

    assert is_escape is True
    # Turning right away from left threat
    assert turn > 0.0
    assert fwd > 0.0


def test_learned_avoidance_memory(test_snn):
    """
    When pain occurs at heading 90 degrees, threat_memory records it.
    Approaching heading 90 degrees should generate an avoidance steering bias away from 90.
    """
    # Initially no danger
    assert test_snn.get_learned_avoidance_steer(90.0) == 0.0

    # Inject pain at world heading 90.0
    test_snn.inject_pain(intensity=1.0, threat_rel_angle=0.0, world_heading=90.0)

    # Approaching 90 degrees now triggers avoidance steer
    steer = test_snn.get_learned_avoidance_steer(90.0)
    assert abs(steer) > 0.0


def test_rain_pain_depolarization_and_dopamine(test_snn):
    """Verify rain pain depolarizes nociceptors and elevates punishment dopamine."""
    assert test_snn.dopamine_level == 0.0
    rain_current = test_snn.inject_rain_pain(intensity=1.0, world_heading=45.0)

    assert rain_current.shape == (test_snn.num_neurons,)
    assert rain_current.sum() > 0.0
    assert test_snn.dopamine_level > 0.3
    assert test_snn.threat_memory[45].item() > 0.2

    spikes = test_snn.step(rain_current)
    assert spikes.sum() > 0


def test_rain_shelter_conditioned_avoidance(test_snn):
    """
    When rain pain is received with shelter on the LEFT (-45 deg from heading 0),
    headings facing AWAY from shelter (opposite heading ~135 deg) are punished in Mushroom Body.
    Approaching the dangerous open direction generates learned avoidance steering away from danger.
    """
    test_snn.inject_rain_pain(intensity=1.0, shelter_rel_angle=-45.0, world_heading=0.0)

    # Heading facing away from shelter is conditioned as dangerous
    steer = test_snn.get_learned_avoidance_steer(135.0)
    assert abs(steer) > 0.0


def test_shelter_relief_suppresses_punishment(test_snn):
    """
    Moving under shelter halts nociceptive pain and releases relief dopamine,
    soothing aversive punishment levels.
    """
    test_snn.inject_rain_pain(intensity=1.0, world_heading=0.0)
    dopamine_after_rain = test_snn.dopamine_level

    relief_current = test_snn.inject_shelter_relief()
    assert relief_current.shape == (test_snn.num_neurons,)
    assert test_snn.dopamine_level < dopamine_after_rain
    assert not test_snn.is_escaping

