"""Unit tests for Chemosensory Food Perception, Gustation, and Chemotaxis."""

import pytest
import torch

from connectome_loader import generate_synthetic_connectome
from snn_engine import FlyBrainSNN


def test_chemosensory_indexing():
    conn = generate_synthetic_connectome(num_visual=64, num_steering=128, num_motor=32)
    assert hasattr(conn, "olfactory_indices")
    assert hasattr(conn, "gustatory_indices")
    assert conn.olfactory_indices is not None
    assert conn.gustatory_indices is not None
    assert len(conn.olfactory_indices) > 0
    assert len(conn.gustatory_indices) > 0


def test_food_odor_injection():
    conn = generate_synthetic_connectome(num_visual=64, num_steering=128, num_motor=32)
    snn = FlyBrainSNN(conn, sensory_gain=30.0)

    # Initial states
    assert snn.hunger_level == 0.5
    assert snn.appetitive_dopamine == 0.0

    # Inject food odor from the left (-45 deg)
    odor_current = snn.inject_food_odor(valence=1.0, rel_angle=-45.0, is_feeding=False)
    assert odor_current.shape == (conn.num_neurons,)
    assert (odor_current > 0).any()

    # Verify left motor neurons got chemotactic steering bias
    left_stim = odor_current[conn.motor_left_indices].sum().item()
    right_stim = odor_current[conn.motor_right_indices].sum().item()
    assert left_stim > right_stim, "Odor to the left should stimulate left motor turning circuits!"


def test_feeding_nutrient_reward():
    conn = generate_synthetic_connectome(num_visual=64, num_steering=128, num_motor=32)
    snn = FlyBrainSNN(conn, sensory_gain=30.0)

    initial_hunger = snn.hunger_level
    odor_current = snn.inject_food_odor(valence=1.0, rel_angle=0.0, is_feeding=True)

    # Feeding should trigger gustatory taste, release appetitive PAM dopamine, and reduce hunger
    assert snn.appetitive_dopamine > 0.0
    assert snn.hunger_level < initial_hunger
    gustatory_stim = odor_current[conn.gustatory_indices].sum().item()
    assert gustatory_stim > 0.0, "Gustatory neurons should be stimulated during active feeding!"
