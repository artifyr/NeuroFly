"""
Tests for Connectome Data Parsing and Sparse Matrix Builder.
"""

import io
import pytest
import torch

from connectome_loader import (
    ConnectomeData,
    build_sparse_weights,
    generate_synthetic_connectome,
    load_connectome_from_csv,
)


def test_build_sparse_weights():
    """Verify building sparse COO tensor with weight normalization."""
    pre_idx = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    post_idx = torch.tensor([1, 2, 2, 0], dtype=torch.int64)
    counts = torch.tensor([5.0, 10.0, 10.0, 4.0], dtype=torch.float32)
    num_neurons = 3

    sparse_w = build_sparse_weights(
        pre_idx, post_idx, counts, num_neurons, device=torch.device("cpu"), normalize=True
    )

    assert sparse_w.is_sparse
    assert sparse_w.shape == (3, 3)
    assert sparse_w._nnz() == 4

    dense = sparse_w.to_dense()
    # Neuron 2 has incoming synapses from neuron 1 (count 10) and neuron 0 (count 10), total 20
    # Normalized weights into neuron 2 should sum to 1.0
    assert torch.isclose(dense[2, :].sum(), torch.tensor(1.0))
    assert torch.isclose(dense[2, 0], torch.tensor(0.5))
    assert torch.isclose(dense[2, 1], torch.tensor(0.5))


def test_load_connectome_from_csv_buffer():
    """Verify loading from CSV text buffer with FAFB schema."""
    csv_data = io.StringIO(
        "pre_root_id,post_root_id,synapse_count\n"
        "1001,1002,15\n"
        "1002,1003,25\n"
        "1003,1001,5\n"
        "1001,1003,10\n"
    )

    annot_data = io.StringIO(
        "root_id,group\n"
        "1001,visual_left\n"
        "1002,steering\n"
        "1003,motor_left\n"
    )

    connectome = load_connectome_from_csv(
        edges_source=csv_data,
        annotations_source=annot_data,
        device="cpu",
    )

    assert connectome.num_neurons == 3
    assert connectome.weight_matrix.is_sparse
    assert connectome.weight_matrix.shape == (3, 3)
    assert connectome.weight_matrix._nnz() == 4
    assert len(connectome.visual_left_indices) == 1
    assert len(connectome.steering_indices) == 1
    assert len(connectome.motor_left_indices) == 1
    assert len(connectome.motor_indices) == 1


def test_synthetic_connectome_generation():
    """Verify generating synthetic connectome across all functional partitions."""
    num_vis = 32
    num_steer = 64
    num_mot = 16
    total_expected = num_vis + num_steer + num_mot

    connectome = generate_synthetic_connectome(
        num_visual=num_vis,
        num_steering=num_steer,
        num_motor=num_mot,
        density=0.08,
        seed=123,
        device="cpu",
    )

    assert connectome.num_neurons == total_expected
    assert connectome.weight_matrix.shape == (total_expected, total_expected)
    assert connectome.weight_matrix.is_sparse

    # Non-zero connections must be populated
    nnz = connectome.weight_matrix._nnz()
    assert nnz > 0, "Sparse matrix should contain non-zero synaptic edges."

    # Partitions
    assert len(connectome.visual_indices) == num_vis
    assert len(connectome.visual_left_indices) == num_vis // 2
    assert len(connectome.visual_right_indices) == num_vis // 2
    assert len(connectome.steering_indices) == num_steer
    assert len(connectome.motor_indices) == num_mot
    assert len(connectome.motor_left_indices) == num_mot // 2
    assert len(connectome.motor_right_indices) == num_mot // 2

    # Verify no overlapping indices between partitions
    vis_set = set(connectome.visual_indices.tolist())
    steer_set = set(connectome.steering_indices.tolist())
    mot_set = set(connectome.motor_indices.tolist())

    assert vis_set.isdisjoint(steer_set)
    assert steer_set.isdisjoint(mot_set)
    assert vis_set.isdisjoint(mot_set)
    assert len(vis_set | steer_set | mot_set) == total_expected


def test_load_sample_fafb_files():
    """Verify loading actual sample CSV files from the data directory."""
    import os
    edges_path = os.path.join("data", "sample_fafb_connectome.csv")
    annot_path = os.path.join("data", "sample_fafb_annotations.csv")

    connectome = load_connectome_from_csv(
        edges_source=edges_path,
        annotations_source=annot_path,
        device="cpu",
    )

    assert connectome.num_neurons == 8
    assert connectome.weight_matrix.shape == (8, 8)
    assert connectome.weight_matrix._nnz() == 10
    assert len(connectome.visual_left_indices) == 2
    assert len(connectome.visual_right_indices) == 2
    assert len(connectome.steering_indices) == 2
    assert len(connectome.motor_left_indices) == 1
    assert len(connectome.motor_right_indices) == 1
