"""
Connectome Ingestion and Sparse Matrix Builder for Drosophila FAFB Data.

Loads synaptic edge lists (pre_root_id, post_root_id, synapse_count), builds
normalized PyTorch sparse COO tensors, and partitions neurons into functional groups:
Visual Input Neurons, Central Complex / Steering Neurons, and Descending Motor Neurons.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class ConnectomeData:
    """Container for parsed connectome structure and functional partitions."""
    weight_matrix: torch.Tensor  # torch.sparse_coo_tensor of shape (N, N)
    num_neurons: int
    neuron_ids: List[int]
    id_to_idx: Dict[int, int]
    visual_indices: torch.Tensor  # All sensory input indices
    visual_left_indices: torch.Tensor
    visual_right_indices: torch.Tensor
    steering_indices: torch.Tensor  # Central complex / intermediate processing
    motor_indices: torch.Tensor  # All descending motor neuron indices
    motor_left_indices: torch.Tensor
    motor_right_indices: torch.Tensor
    device: torch.device
    nociceptive_indices: Optional[torch.Tensor] = None  # Mechanosensory & thermosensory pain/touch neurons
    dopamine_indices: Optional[torch.Tensor] = None  # DAN dopaminergic punishment neurons (PPL1 cluster)
    kenyon_indices: Optional[torch.Tensor] = None  # Kenyon cells (Mushroom Body associative memory)
    mbon_indices: Optional[torch.Tensor] = None  # MBON (Mushroom Body Output Neurons approach/avoidance)
    olfactory_indices: Optional[torch.Tensor] = None  # Odor receptors and antennal lobe projection neurons (ALPN)
    gustatory_indices: Optional[torch.Tensor] = None  # Taste receptors (sweet, bitter, organic)
    thermosensory_indices: Optional[torch.Tensor] = None  # Thermal nociceptors (fire, lava, heat avoidance)
    ocellar_indices: Optional[torch.Tensor] = None  # Dorsal ocellar photoreceptors for phototaxis (torches, sunlight)
    mechanosensory_indices: Optional[torch.Tensor] = None  # Bristle sensilla for perching & grooming programs
    circadian_indices: Optional[torch.Tensor] = None  # Pars intercerebralis & lateralis for sleep/quiescent cycles

    def __post_init__(self) -> None:
        if self.nociceptive_indices is None:
            self.nociceptive_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.dopamine_indices is None:
            self.dopamine_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.kenyon_indices is None:
            self.kenyon_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.mbon_indices is None:
            self.mbon_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.olfactory_indices is None:
            self.olfactory_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.gustatory_indices is None:
            self.gustatory_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.thermosensory_indices is None:
            self.thermosensory_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.ocellar_indices is None:
            self.ocellar_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.mechanosensory_indices is None:
            self.mechanosensory_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.circadian_indices is None:
            self.circadian_indices = torch.empty((0,), dtype=torch.int64, device=self.device)

    def to(self, device: Union[torch.device, str]) -> "ConnectomeData":
        """Transfer tensors to the target device."""
        target_dev = torch.device(device)
        return ConnectomeData(
            weight_matrix=self.weight_matrix.to(target_dev),
            num_neurons=self.num_neurons,
            neuron_ids=self.neuron_ids,
            id_to_idx=self.id_to_idx,
            visual_indices=self.visual_indices.to(target_dev),
            visual_left_indices=self.visual_left_indices.to(target_dev),
            visual_right_indices=self.visual_right_indices.to(target_dev),
            steering_indices=self.steering_indices.to(target_dev),
            motor_indices=self.motor_indices.to(target_dev),
            motor_left_indices=self.motor_left_indices.to(target_dev),
            motor_right_indices=self.motor_right_indices.to(target_dev),
            device=target_dev,
            nociceptive_indices=self.nociceptive_indices.to(target_dev),
            dopamine_indices=self.dopamine_indices.to(target_dev),
            kenyon_indices=self.kenyon_indices.to(target_dev),
            mbon_indices=self.mbon_indices.to(target_dev),
            olfactory_indices=self.olfactory_indices.to(target_dev),
            gustatory_indices=self.gustatory_indices.to(target_dev),
            thermosensory_indices=self.thermosensory_indices.to(target_dev),
            ocellar_indices=self.ocellar_indices.to(target_dev),
            mechanosensory_indices=self.mechanosensory_indices.to(target_dev),
            circadian_indices=self.circadian_indices.to(target_dev),
        )


def build_sparse_weights(
    pre_indices: torch.Tensor,
    post_indices: torch.Tensor,
    synapse_counts: torch.Tensor,
    num_neurons: int,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build a PyTorch sparse COO matrix of shape (num_neurons, num_neurons).
    
    Weights are indexed as W[post, pre], such that post-synaptic current
    is computed as I_post = W @ spikes_pre.

    Parameters
    ----------
    pre_indices : torch.Tensor
        1D tensor of source neuron indices (int64).
    post_indices : torch.Tensor
        1D tensor of target neuron indices (int64).
    synapse_counts : torch.Tensor
        1D tensor of synaptic counts / connection strengths (float32).
    num_neurons : int
        Total number of neurons N.
    device : torch.device
        Target device for the sparse tensor.
    normalize : bool
        If True, incoming synaptic weights for each post-synaptic neuron
        are normalized by the total incoming synapse count of that neuron.
    """
    if len(pre_indices) == 0:
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (num_neurons, num_neurons), device=device, check_invariants=False)

    pre_indices = pre_indices.to(dtype=torch.int64, device=device)
    post_indices = post_indices.to(dtype=torch.int64, device=device)
    counts = synapse_counts.to(dtype=torch.float32, device=device)

    if normalize:
        # Sum incoming synapse counts for each post-synaptic neuron
        in_degrees = torch.zeros(num_neurons, dtype=torch.float32, device=device)
        in_degrees.scatter_add_(0, post_indices, counts)
        in_degrees_clamped = torch.clamp(in_degrees, min=1.0)
        norm_factor = in_degrees_clamped[post_indices]
        weights = counts / norm_factor
    else:
        weights = counts

    # Indices tensor: row 0 is post (target), row 1 is pre (source)
    indices = torch.stack([post_indices, pre_indices], dim=0)
    sparse_weights = torch.sparse_coo_tensor(
        indices, weights, (num_neurons, num_neurons), device=device, check_invariants=False
    ).coalesce()

    return sparse_weights


def load_connectome_from_csv(
    edges_source: Union[str, io.StringIO, pd.DataFrame],
    annotations_source: Optional[Union[str, io.StringIO, pd.DataFrame]] = None,
    device: Union[torch.device, str] = "cpu",
    normalize: bool = True,
) -> ConnectomeData:
    """
    Ingest FAFB connectome edge list and annotations.

    Edge list schema:
      - pre_root_id: int / int64
      - post_root_id: int / int64
      - synapse_count: int / float

    Optional annotations schema:
      - root_id: int / int64
      - group: str in {'visual_left', 'visual_right', 'visual', 'steering', 'motor_left', 'motor_right', 'motor'}
    """
    target_device = torch.device(device)

    # Ingest edges DataFrame
    if isinstance(edges_source, pd.DataFrame):
        df_edges = edges_source.copy()
    elif isinstance(edges_source, io.StringIO):
        df_edges = pd.read_csv(edges_source)
    else:
        df_edges = pd.read_csv(edges_source)

    # Normalize column names (support FlyWire 'syn_count' or 'synapse_count')
    col_mapping = {
        "syn_count": "synapse_count",
        "pre": "pre_root_id",
        "post": "post_root_id",
        "weight": "synapse_count",
    }
    df_edges.rename(columns=col_mapping, inplace=True)

    required_cols = {"pre_root_id", "post_root_id", "synapse_count"}
    if not required_cols.issubset(set(df_edges.columns)):
        raise ValueError(f"Edges CSV missing required columns: {required_cols - set(df_edges.columns)}")

    # Clean & filter
    df_edges = df_edges[df_edges["synapse_count"] > 0].copy()

    # Collect unique neuron IDs
    unique_pre = df_edges["pre_root_id"].unique()
    unique_post = df_edges["post_root_id"].unique()
    all_unique_ids = sorted(list(set(unique_pre).union(set(unique_post))))

    num_neurons = len(all_unique_ids)
    id_to_idx = {nid: idx for idx, nid in enumerate(all_unique_ids)}

    pre_idx = torch.tensor([id_to_idx[nid] for nid in df_edges["pre_root_id"]], dtype=torch.int64)
    post_idx = torch.tensor([id_to_idx[nid] for nid in df_edges["post_root_id"]], dtype=torch.int64)
    syn_counts = torch.tensor(df_edges["synapse_count"].values, dtype=torch.float32)

    sparse_weights = build_sparse_weights(
        pre_idx, post_idx, syn_counts, num_neurons, device=target_device, normalize=normalize
    )

    # Process annotations or default partitioning
    visual_left_list: List[int] = []
    visual_right_list: List[int] = []
    steering_list: List[int] = []
    motor_left_list: List[int] = []
    motor_right_list: List[int] = []
    nociceptive_list: List[int] = []
    dopamine_list: List[int] = []
    kenyon_list: List[int] = []
    mbon_list: List[int] = []
    olfactory_list: List[int] = []
    gustatory_list: List[int] = []
    thermosensory_list: List[int] = []
    ocellar_list: List[int] = []
    mechanosensory_list: List[int] = []
    circadian_list: List[int] = []

    if annotations_source is not None:
        if isinstance(annotations_source, pd.DataFrame):
            df_annot = annotations_source.copy()
        elif isinstance(annotations_source, io.StringIO):
            df_annot = pd.read_csv(annotations_source)
        else:
            df_annot = pd.read_csv(annotations_source)

        # Check if this is the official FlyWire classification schema:
        # columns: root_id, super_class, side, flow, class, etc.
        has_flywire_schema = "super_class" in df_annot.columns

        for _, row in df_annot.iterrows():
            nid = row.get("root_id")
            if nid not in id_to_idx:
                continue
            idx = id_to_idx[nid]

            if has_flywire_schema:
                super_class = str(row.get("super_class", "")).lower().strip()
                side = str(row.get("side", "")).lower().strip()
                n_class = str(row.get("class", "")).strip()
                sub_class = str(row.get("sub_class", "")).strip()

                # Specific functional types
                if n_class in ("mechanosensory", "thermosensory"):
                    nociceptive_list.append(idx)
                if n_class == "thermosensory":
                    thermosensory_list.append(idx)
                elif n_class == "ocellar":
                    ocellar_list.append(idx)
                elif n_class == "mechanosensory" or sub_class == "eye_bristle":
                    mechanosensory_list.append(idx)
                elif n_class in ("pars_intercerebralis", "pars_lateralis"):
                    circadian_list.append(idx)
                elif n_class == "DAN":
                    dopamine_list.append(idx)
                elif n_class == "Kenyon_Cell":
                    kenyon_list.append(idx)
                elif n_class == "MBON":
                    mbon_list.append(idx)
                elif n_class in ("olfactory", "ALPN"):
                    olfactory_list.append(idx)
                elif n_class == "gustatory":
                    gustatory_list.append(idx)

                if super_class in ("optic", "sensory", "visual_projection"):
                    if side == "left":
                        visual_left_list.append(idx)
                    elif side == "right":
                        visual_right_list.append(idx)
                    else:
                        if len(visual_left_list) <= len(visual_right_list):
                            visual_left_list.append(idx)
                        else:
                            visual_right_list.append(idx)
                elif super_class in ("central", "central_complex", "ascending", "sensory_ascending"):
                    steering_list.append(idx)
                elif super_class in ("descending", "motor"):
                    if side == "left":
                        motor_left_list.append(idx)
                    elif side == "right":
                        motor_right_list.append(idx)
                    else:
                        if len(motor_left_list) <= len(motor_right_list):
                            motor_left_list.append(idx)
                        else:
                            motor_right_list.append(idx)
            else:
                # Custom group format
                group = str(row.get("group", "")).lower().strip()
                if group in ("visual_left", "eye_left", "optic_left"):
                    visual_left_list.append(idx)
                elif group in ("visual_right", "eye_right", "optic_right"):
                    visual_right_list.append(idx)
                elif group in ("visual", "sensory"):
                    if len(visual_left_list) <= len(visual_right_list):
                        visual_left_list.append(idx)
                    else:
                        visual_right_list.append(idx)
                elif group in ("steering", "central_complex", "cx", "intermediate"):
                    steering_list.append(idx)
                elif group in ("motor_left", "descending_left", "dn_left"):
                    motor_left_list.append(idx)
                elif group in ("motor_right", "descending_right", "dn_right"):
                    motor_right_list.append(idx)
                elif group in ("motor", "descending", "dn"):
                    if len(motor_left_list) <= len(motor_right_list):
                        motor_left_list.append(idx)
                    else:
                        motor_right_list.append(idx)
                elif group in ("nociceptive", "pain"):
                    nociceptive_list.append(idx)
                elif group in ("thermosensory", "thermal", "heat"):
                    thermosensory_list.append(idx)
                    nociceptive_list.append(idx)
                elif group in ("ocellar", "ocelli", "light"):
                    ocellar_list.append(idx)
                elif group in ("mechanosensory", "grooming", "bristle"):
                    mechanosensory_list.append(idx)
                elif group in ("circadian", "sleep", "clock"):
                    circadian_list.append(idx)
                elif group in ("dopamine", "dan"):
                    dopamine_list.append(idx)
                elif group in ("kenyon", "kenyon_cell", "mb"):
                    kenyon_list.append(idx)
                elif group in ("mbon",):
                    mbon_list.append(idx)
                elif group in ("olfactory", "smell", "odor"):
                    olfactory_list.append(idx)
                elif group in ("gustatory", "taste"):
                    gustatory_list.append(idx)

    # If annotations were missing or incomplete, partition based on index proportions
    if not (visual_left_list or visual_right_list or steering_list or motor_left_list or motor_right_list):
        n_vis = max(2, int(num_neurons * 0.2))
        n_mot = max(2, int(num_neurons * 0.2))
        n_steer = max(1, num_neurons - n_vis - n_mot)

        vis_half = n_vis // 2
        visual_left_list = list(range(0, vis_half))
        visual_right_list = list(range(vis_half, n_vis))

        steering_list = list(range(n_vis, n_vis + n_steer))

        mot_start = n_vis + n_steer
        mot_half = n_mot // 2
        motor_left_list = list(range(mot_start, mot_start + mot_half))
        motor_right_list = list(range(mot_start + mot_half, num_neurons))

    # Biological fallback for specialized groups if not found in custom format
    if not nociceptive_list:
        nociceptive_list = visual_left_list[:max(1, len(visual_left_list) // 4)]
    if not thermosensory_list:
        thermosensory_list = nociceptive_list[:max(1, len(nociceptive_list) // 8)]
    if not ocellar_list:
        ocellar_list = visual_left_list[:max(1, len(visual_left_list) // 10)]
    if not mechanosensory_list:
        mechanosensory_list = nociceptive_list
    if not circadian_list:
        circadian_list = steering_list[:max(1, len(steering_list) // 20)]
    if not dopamine_list:
        dopamine_list = steering_list[:max(1, len(steering_list) // 10)]
    if not kenyon_list:
        kenyon_list = steering_list[len(dopamine_list):len(dopamine_list) + max(1, len(steering_list) // 4)]
    if not mbon_list:
        mbon_list = steering_list[len(dopamine_list) + len(kenyon_list):len(dopamine_list) + len(kenyon_list) + max(1, len(steering_list) // 8)]
    if not olfactory_list:
        olfactory_list = visual_right_list[:max(1, len(visual_right_list) // 4)]
    if not gustatory_list:
        gustatory_list = visual_right_list[len(olfactory_list):len(olfactory_list) + max(1, len(visual_right_list) // 8)]

    # Convert to tensors
    v_l = torch.tensor(visual_left_list, dtype=torch.int64, device=target_device)
    v_r = torch.tensor(visual_right_list, dtype=torch.int64, device=target_device)
    v_all = torch.tensor(sorted(visual_left_list + visual_right_list), dtype=torch.int64, device=target_device)

    steer = torch.tensor(sorted(steering_list), dtype=torch.int64, device=target_device)

    m_l = torch.tensor(motor_left_list, dtype=torch.int64, device=target_device)
    m_r = torch.tensor(motor_right_list, dtype=torch.int64, device=target_device)
    m_all = torch.tensor(sorted(motor_left_list + motor_right_list), dtype=torch.int64, device=target_device)

    noci = torch.tensor(sorted(nociceptive_list), dtype=torch.int64, device=target_device)
    thermo = torch.tensor(sorted(thermosensory_list), dtype=torch.int64, device=target_device)
    ocel = torch.tensor(sorted(ocellar_list), dtype=torch.int64, device=target_device)
    mech = torch.tensor(sorted(mechanosensory_list), dtype=torch.int64, device=target_device)
    circ = torch.tensor(sorted(circadian_list), dtype=torch.int64, device=target_device)
    dopa = torch.tensor(sorted(dopamine_list), dtype=torch.int64, device=target_device)
    keny = torch.tensor(sorted(kenyon_list), dtype=torch.int64, device=target_device)
    mbon = torch.tensor(sorted(mbon_list), dtype=torch.int64, device=target_device)
    olfa = torch.tensor(sorted(olfactory_list), dtype=torch.int64, device=target_device)
    gust = torch.tensor(sorted(gustatory_list), dtype=torch.int64, device=target_device)

    return ConnectomeData(
        weight_matrix=sparse_weights,
        num_neurons=num_neurons,
        neuron_ids=all_unique_ids,
        id_to_idx=id_to_idx,
        visual_indices=v_all,
        visual_left_indices=v_l,
        visual_right_indices=v_r,
        steering_indices=steer,
        motor_indices=m_all,
        motor_left_indices=m_l,
        motor_right_indices=m_r,
        device=target_device,
        nociceptive_indices=noci,
        dopamine_indices=dopa,
        kenyon_indices=keny,
        mbon_indices=mbon,
        olfactory_indices=olfa,
        gustatory_indices=gust,
        thermosensory_indices=thermo,
        ocellar_indices=ocel,
        mechanosensory_indices=mech,
        circadian_indices=circ,
    )


def generate_synthetic_connectome(
    num_visual: int = 64,
    num_steering: int = 128,
    num_motor: int = 32,
    density: float = 0.05,
    seed: int = 42,
    device: Union[torch.device, str] = "cpu",
    normalize: bool = True,
) -> ConnectomeData:
    """
    Generate a biologically plausible synthetic connectome partitioned into:
    - Visual input neurons (Left and Right halves)
    - Central Complex / Steering intermediate recurrent circuits
    - Descending motor neurons (Left and Right motor command pools)
    """
    rng = np.random.default_rng(seed)
    num_neurons = num_visual + num_steering + num_motor

    # Indices
    vis_half = num_visual // 2
    mot_half = num_motor // 2

    visual_left = list(range(0, vis_half))
    visual_right = list(range(vis_half, num_visual))
    steering = list(range(num_visual, num_visual + num_steering))
    motor_left = list(range(num_visual + num_steering, num_visual + num_steering + mot_half))
    motor_right = list(range(num_visual + num_steering + mot_half, num_neurons))

    pre_list: List[int] = []
    post_list: List[int] = []
    syn_list: List[float] = []

    def add_synapses(sources: List[int], targets: List[int], prob: float, min_syn: int = 1, max_syn: int = 10):
        for s in sources:
            for t in targets:
                if s != t and rng.random() < prob:
                    pre_list.append(s)
                    post_list.append(t)
                    syn_list.append(float(rng.integers(min_syn, max_syn + 1)))

    # 1. Visual to Steering (sensory feedforward with lateral bias)
    steer_half = num_steering // 2
    steer_left = steering[:steer_half]
    steer_right = steering[steer_half:]

    # Left visual projects predominantly to ipsilateral / steering circuits
    add_synapses(visual_left, steer_left, prob=density * 2.5, min_syn=2, max_syn=12)
    add_synapses(visual_right, steer_right, prob=density * 2.5, min_syn=2, max_syn=12)
    # Contralateral cross-talk
    add_synapses(visual_left, steer_right, prob=density * 0.8, min_syn=1, max_syn=5)
    add_synapses(visual_right, steer_left, prob=density * 0.8, min_syn=1, max_syn=5)

    # 2. Steering recurrent connections (Central Complex internal dynamics)
    add_synapses(steering, steering, prob=density * 1.5, min_syn=1, max_syn=8)

    # 3. Steering to Descending Motor Neurons
    add_synapses(steer_left, motor_left, prob=density * 2.0, min_syn=2, max_syn=15)
    add_synapses(steer_right, motor_right, prob=density * 2.0, min_syn=2, max_syn=15)
    add_synapses(steer_left, motor_right, prob=density * 0.5, min_syn=1, max_syn=5)
    add_synapses(steer_right, motor_left, prob=density * 0.5, min_syn=1, max_syn=5)

    # 4. Mutual inhibition between motor left and right for crisp steering
    add_synapses(motor_left, motor_right, prob=density * 0.5, min_syn=1, max_syn=4)
    add_synapses(motor_right, motor_left, prob=density * 0.5, min_syn=1, max_syn=4)

    # Build DataFrame format
    df_synthetic = pd.DataFrame({
        "pre_root_id": [1000 + p for p in pre_list],
        "post_root_id": [1000 + p for p in post_list],
        "synapse_count": syn_list,
    })

    # Build annotations DataFrame
    annot_rows = []
    for idx in visual_left:
        annot_rows.append({"root_id": 1000 + idx, "group": "visual_left"})
    for idx in visual_right:
        annot_rows.append({"root_id": 1000 + idx, "group": "visual_right"})
    for idx in steering:
        annot_rows.append({"root_id": 1000 + idx, "group": "steering"})
    for idx in motor_left:
        annot_rows.append({"root_id": 1000 + idx, "group": "motor_left"})
    for idx in motor_right:
        annot_rows.append({"root_id": 1000 + idx, "group": "motor_right"})
    df_annot = pd.DataFrame(annot_rows)

    return load_connectome_from_csv(
        edges_source=df_synthetic,
        annotations_source=df_annot,
        device=device,
        normalize=normalize,
    )
