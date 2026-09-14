"""
Spiking Neural Network Simulator for Drosophila Connectome.

Implements a Leaky Integrate-and-Fire (LIF) network (`FlyBrainSNN`) with
resting potential, threshold detection, refractory clamping, membrane potential
decay, and sparse synaptic propagation across connectome edge weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from connectome_loader import ConnectomeData


class FlyBrainSNN(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neural network engine driven by
    a connectome sparse synaptic weight matrix.

    Parameters
    ----------
    connectome : ConnectomeData
        Parsed connectome structure containing the sparse weight matrix and functional indices.
    v_rest : float
        Resting membrane potential in mV (default: -70.0 mV).
    v_thresh : float
        Action potential threshold in mV (default: -50.0 mV).
    v_reset : float
        Post-spike reset potential in mV (default: -70.0 mV).
    tau : float
        Membrane time constant in ms (default: 20.0 ms).
    dt : float
        Simulation timestep in ms (default: 1.0 ms).
    refractory_steps : int
        Number of timesteps a neuron remains refractory after spiking (default: 2 steps).
    synaptic_gain : float
        Scaling factor applied to incoming synaptic currents (default: 25.0 mV).
    sensory_gain : float
        Scaling factor applied to injected sensory stimuli (default: 30.0 mV).
    """

    def __init__(
        self,
        connectome: ConnectomeData,
        v_rest: float = -70.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
        tau: float = 20.0,
        dt: float = 1.0,
        refractory_steps: int = 2,
        synaptic_gain: float = 25.0,
        sensory_gain: float = 30.0,
    ) -> None:
        super().__init__()
        self.connectome = connectome
        self.num_neurons = connectome.num_neurons
        self.device = connectome.device

        self.v_rest = float(v_rest)
        self.v_thresh = float(v_thresh)
        self.v_reset = float(v_reset)
        self.base_v_thresh = float(v_thresh)
        self.tau = float(tau)
        self.dt = float(dt)
        self.refractory_steps = int(refractory_steps)
        self.synaptic_gain = float(synaptic_gain)
        self.base_synaptic_gain = float(synaptic_gain)
        self.sensory_gain = float(sensory_gain)

        # Discrete decay factor beta = exp(-dt / tau)
        self.beta = float(torch.exp(-torch.tensor(self.dt / self.tau)).item())

        # Register sparse weights as a buffer
        self.register_buffer("weights", connectome.weight_matrix)

        # Dynamic state vectors
        self.register_buffer("v", torch.full((self.num_neurons,), self.v_rest, dtype=torch.float32, device=self.device))
        self.register_buffer("refractory_timer", torch.zeros((self.num_neurons,), dtype=torch.int32, device=self.device))
        self.register_buffer("synaptic_current", torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device))

        # Biological neuromodulation & plasticity states
        self.dopamine_level = 0.0  # PPL1 dopamine concentration [0.0, 1.0] (punishment/pain)
        self.hunger_level = 0.5  # Internal metabolic hunger state [0.0, 1.0]
        self.appetitive_dopamine = 0.0  # PAM cluster dopamine (food reward) [0.0, 1.0]
        self.escape_timer = 0  # Giant Fiber escape reflex steps
        self.is_sleeping = False  # Circadian quiescent rest state
        self.is_grooming = False  # Mechanosensory perching & grooming state
        # Ring attractor memory (360 degrees) of conditioned danger headings
        self.register_buffer("threat_memory", torch.zeros(360, dtype=torch.float32, device=self.device))

    def reset_state(self) -> None:
        """Reset all dynamic states (voltages, refractory counters, synaptic currents, neuromodulators)."""
        self.v.fill_(self.v_rest)
        self.refractory_timer.zero_()
        self.synaptic_current.zero_()
        self.dopamine_level = 0.0
        self.hunger_level = 0.5
        self.appetitive_dopamine = 0.0
        self.escape_timer = 0
        self.is_sleeping = False
        self.is_grooming = False
        self.v_thresh = self.base_v_thresh
        self.synaptic_gain = self.base_synaptic_gain
        self.threat_memory.zero_()

    def set_device(self, device: Union[torch.device, str]) -> "FlyBrainSNN":
        """Move the SNN and connectome buffers to the given device."""
        target_device = torch.device(device)
        self.connectome = self.connectome.to(target_device)
        self.device = target_device
        self.to(target_device)
        return self

    def step(self, external_current: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance the simulation by a single discrete timestep dt.

        Parameters
        ----------
        external_current : torch.Tensor, optional
            External current vector of shape (num_neurons,) injected directly.

        Returns
        -------
        spikes : torch.Tensor
            Binary spike vector of shape (num_neurons,) where 1.0 indicates a spike.
        """
        # 1. Identify currently refractory neurons
        is_refractory = self.refractory_timer > 0

        # 2. Update membrane potential for non-refractory neurons:
        # V(t) = V_rest + (V(t-1) - V_rest) * beta + I_syn + I_ext
        total_input = self.synaptic_current
        if external_current is not None:
            total_input = total_input + external_current.to(self.device)

        decayed_v = self.v_rest + (self.v - self.v_rest) * self.beta + total_input
        self.v = torch.where(is_refractory, torch.full_like(self.v, self.v_reset), decayed_v)

        # 3. Detect spikes: V >= V_thresh for non-refractory neurons
        can_fire = ~is_refractory
        spikes_mask = can_fire & (self.v >= self.v_thresh)
        spikes = spikes_mask.to(dtype=torch.float32)

        # 4. Post-spike reset and set refractory timer
        self.v = torch.where(spikes_mask, torch.full_like(self.v, self.v_reset), self.v)
        self.refractory_timer = torch.where(
            spikes_mask,
            torch.full_like(self.refractory_timer, self.refractory_steps),
            torch.clamp(self.refractory_timer - 1, min=0),
        )

        # 5. Propagate action potentials across sparse connectivity matrix
        # I_syn = synaptic_gain * (Weights @ spikes)
        # Note: Weights is of shape (N, N) where row=post, col=pre
        if spikes.any():
            spikes_col = spikes.unsqueeze(1)  # (N, 1)
            # torch.sparse.mm handles sparse @ dense -> dense (N, 1)
            propagated = torch.sparse.mm(self.weights, spikes_col).squeeze(1)
            self.synaptic_current = propagated * self.synaptic_gain
        else:
            self.synaptic_current.zero_()

        # 6. Neuromodulatory clearance & memory decay
        self.dopamine_level *= 0.96
        self.threat_memory *= 0.995
        if self.escape_timer > 0:
            self.escape_timer -= 1

        return spikes

    def inject_pain(self, intensity: float = 1.0, threat_rel_angle: float = 0.0, world_heading: float = 0.0) -> torch.Tensor:
        """
        Inject high-frequency nociceptive shock into the connectome:
        - Fires mechanosensory/nociceptive neurons.
        - Activates DAN dopaminergic punishment neurons.
        - Triggers the Giant Fiber escape reflex turning AWAY from the threat.
        - Encodes threat heading into Mushroom Body memory.
        """
        pain_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)
        self.dopamine_level = min(1.0, self.dopamine_level + intensity * 0.85)
        self.escape_timer = 16  # Active for ~16 ticks (0.8s escape sequence)
        if self.is_sleeping:
            self.is_sleeping = False
            self.v_thresh = self.base_v_thresh
            self.synaptic_gain = self.base_synaptic_gain

        # 1. Nociceptive sensory neurons (extreme depolarizing surge)
        if len(self.connectome.nociceptive_indices) > 0:
            pain_current[self.connectome.nociceptive_indices] = 55.0 * intensity

        # 2. DAN (Dopaminergic punishment neurons)
        if len(self.connectome.dopamine_indices) > 0:
            pain_current[self.connectome.dopamine_indices] = 45.0 * intensity

        # 3. Giant Fiber escape motor reflex (contralateral jump-turn away from threat)
        # Base forward motor activation (sub-threshold boost)
        if len(self.connectome.motor_indices) > 0:
            pain_current[self.connectome.motor_indices] += 12.0 * intensity

        # If threat is on the left (threat_rel_angle < 0), fire right motor neurons to bank right sharply!
        if threat_rel_angle < 0:
            if len(self.connectome.motor_right_indices) > 0:
                pain_current[self.connectome.motor_right_indices] += 30.0 * intensity
        else:
            if len(self.connectome.motor_left_indices) > 0:
                pain_current[self.connectome.motor_left_indices] += 30.0 * intensity

        # 4. Record dangerous heading into Mushroom Body threat memory
        h_deg = int(world_heading) % 360
        for d in range(-35, 36):
            idx = (h_deg + d) % 360
            self.threat_memory[idx] = min(1.0, self.threat_memory[idx].item() + 0.7 * intensity)

        return pain_current

    def inject_food_odor(
        self,
        valence: float = 1.0,
        rel_angle: float = 0.0,
        is_feeding: bool = False,
    ) -> torch.Tensor:
        """
        Biological Chemosensory & Food Attraction Stimulus:
        - Depolarizes Olfactory Receptor Neurons & ALPNs proportional to odor concentration.
        - If in physical contact (<2.2m), activates Gustatory taste receptor neurons.
        - Injects positive steering current toward food direction (positive chemotaxis).
        - Releases PAM appetitive dopamine reward, satiating hunger and soothing pain.

        Parameters
        ----------
        valence : float
            Odor strength and attractiveness [0.0, 1.0] (e.g. rotten flesh = 1.0, sugar = 0.75).
        rel_angle : float
            Relative angle to food source (-180 to +180 deg).
        is_feeding : bool
            True if fly is within feeding proximity.
        """
        odor_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)
        # Biological Neuropeptide F (sNPF/NPF) sensitization: hunger scales odor drive by up to 3.5x
        hunger_boost = 1.0 + 2.5 * max(0.0, min(1.0, self.hunger_level))
        effective_drive = valence * (0.6 + 0.4 * self.hunger_level) * hunger_boost

        # 1. Olfactory Sensory Projection Activation
        olfa_idx = getattr(self.connectome, "olfactory_indices", None)
        if olfa_idx is not None and len(olfa_idx) > 0:
            odor_current[olfa_idx] = 40.0 * min(3.5, effective_drive)

        # 2. Gustatory Taste Activation (if feeding)
        gust_idx = getattr(self.connectome, "gustatory_indices", None)
        if is_feeding and gust_idx is not None and len(gust_idx) > 0:
            odor_current[gust_idx] = 50.0 * effective_drive

        # 3. Appetitive PAM Dopamine Reward
        if is_feeding:
            self.appetitive_dopamine = min(1.0, self.appetitive_dopamine + 0.08 * valence)
            self.hunger_level = max(0.05, self.hunger_level - 0.02)  # Satiation
            self.dopamine_level = max(0.0, self.dopamine_level - 0.05)  # Food suppresses stress/pain
        else:
            # Appetitive drive increases hunger slightly while searching
            self.hunger_level = min(1.0, self.hunger_level + 0.0005)

        # 4. Chemotactic Steering: Bias descending motor neurons toward food
        # If food is to the left (rel_angle < 0), bias left motor neurons to bank left toward food!
        steer_strength = 24.0 * effective_drive
        if rel_angle < 0:
            if len(self.connectome.motor_left_indices) > 0:
                odor_current[self.connectome.motor_left_indices] += steer_strength
        else:
            if len(self.connectome.motor_right_indices) > 0:
                odor_current[self.connectome.motor_right_indices] += steer_strength

        # Positive forward flight drive toward food source
        if len(self.connectome.motor_indices) > 0:
            odor_current[self.connectome.motor_indices] += 10.0 * effective_drive

        return odor_current

    def inject_thermal_hazard(self, intensity: float = 1.0, rel_angle: float = 0.0) -> torch.Tensor:
        """
        Inject extreme thermal nociception from fire/lava/campfires:
        - Fires thermosensory TRPA1 neurons.
        - Spikes aversive dopamine (PPL1 cluster).
        - Direct evasive steering current away from heat source.
        """
        thermal_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)
        self.dopamine_level = min(1.0, self.dopamine_level + intensity * 0.40)

        # 1. Thermosensory neuron depolarization
        thermo_idx = getattr(self.connectome, "thermosensory_indices", None)
        if thermo_idx is not None and len(thermo_idx) > 0:
            thermal_current[thermo_idx] = 55.0 * intensity

        # Also activate general nociceptive population
        noci_idx = getattr(self.connectome, "nociceptive_indices", None)
        if noci_idx is not None and len(noci_idx) > 0:
            thermal_current[noci_idx] = 30.0 * intensity

        # 2. Steer AWAY from the heat source (if heat is left, turn right)
        steer_strength = 28.0 * intensity
        if rel_angle < 0:
            # Heat is on left -> drive RIGHT motor neurons to turn right
            if len(self.connectome.motor_right_indices) > 0:
                thermal_current[self.connectome.motor_right_indices] += steer_strength
        else:
            # Heat is on right -> drive LEFT motor neurons to turn left
            if len(self.connectome.motor_left_indices) > 0:
                thermal_current[self.connectome.motor_left_indices] += steer_strength

        # Evasive forward boost away from heat
        if len(self.connectome.motor_indices) > 0:
            thermal_current[self.connectome.motor_indices] += 12.0 * intensity

        return thermal_current

    def inject_light_stimulus(self, intensity: float = 1.0, rel_angle: float = 0.0) -> torch.Tensor:
        """
        Positive phototaxis towards torches, lanterns, and sunlit sky:
        - Depolarizes dorsal ocelli (ocellar_indices) and optic lobe photoreceptors.
        - Steers TOWARDS the light source.
        """
        light_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)

        # 1. Ocellar photoreceptor stimulation
        ocel_idx = getattr(self.connectome, "ocellar_indices", None)
        if ocel_idx is not None and len(ocel_idx) > 0:
            light_current[ocel_idx] = 35.0 * intensity

        # 2. Asymmetric optical flow bias towards light
        steer_strength = 18.0 * intensity
        if rel_angle < 0:
            # Light is on left -> drive left motor neurons to bank left towards light
            if len(self.connectome.motor_left_indices) > 0:
                light_current[self.connectome.motor_left_indices] += steer_strength
        else:
            # Light is on right -> drive right motor neurons to bank right towards light
            if len(self.connectome.motor_right_indices) > 0:
                light_current[self.connectome.motor_right_indices] += steer_strength

        if len(self.connectome.motor_indices) > 0:
            light_current[self.connectome.motor_indices] += 6.0 * intensity

        return light_current

    def inject_mechanosensory_grooming(self) -> torch.Tensor:
        """
        Activates tactile bristle mechanoreceptors when landed,
        initiating the stereotyped grooming motor program (antennae, eye, and wing sweeps).
        """
        groom_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)
        self.is_grooming = True

        mech_idx = getattr(self.connectome, "mechanosensory_indices", None)
        if mech_idx is not None and len(mech_idx) > 0:
            groom_current[mech_idx] = 32.0

        # Mechanosensory feedback promotes contentment (suppresses pain)
        self.dopamine_level = max(0.0, self.dopamine_level - 0.02)
        return groom_current

    def inject_johnstons_organ_sound_touch(self, intensity: float = 1.0) -> torch.Tensor:
        """
        Johnston's Organ & Chordotonal Mechanoreceptor Activation (Sound as Physical Touch):
        In Drosophila, auditory vibrations (nearby footsteps, note blocks, block breaking within 1.5m)
        are transduced by Johnston's organ chordotonal neurons, sharing mechanosensory pathways with touch.
        Depolarizes chordotonal mechanoreceptors, triggers flinch response, and startle escape flight.
        """
        mech_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)
        mech_idx = getattr(self.connectome, "mechanosensory_indices", None)
        if mech_idx is not None and len(mech_idx) > 0:
            mech_current[mech_idx] = 48.0 * intensity

        # Nociceptive tactile startled flinch
        noci_idx = getattr(self.connectome, "nociceptive_indices", None)
        if noci_idx is not None and len(noci_idx) > 0:
            mech_current[noci_idx] = 18.0 * intensity

        # Motor flight reflex takeoff activation
        if len(self.connectome.motor_indices) > 0:
            mech_current[self.connectome.motor_indices] += 16.0 * intensity

        return mech_current


    def set_circadian_phase(self, is_night: bool) -> None:
        """
        Modulates circadian neurosecretory centers (pars intercerebralis & lateralis).
        During night/darkness without threats, enters low-metabolic quiescent sleep.
        """
        if is_night and not self.is_escaping and self.hunger_level < 0.85:
            self.is_sleeping = True
            # Sleep-promoting hyperpolarization & GABAergic synaptic attenuation
            self.v_thresh = self.base_v_thresh + 6.0
            self.synaptic_gain = self.base_synaptic_gain * 0.40
        else:
            self.is_sleeping = False
            self.v_thresh = self.base_v_thresh
            self.synaptic_gain = self.base_synaptic_gain

    def get_learned_avoidance_steer(self, current_heading: float) -> float:
        """
        Query Mushroom Body threat memory:
        If current heading points toward a conditioned danger zone,
        returns steering bias to turn away from it.
        """
        h_idx = int(current_heading) % 360
        danger_ahead = float(self.threat_memory[h_idx].item())
        if danger_ahead > 0.12:
            left_danger = float(self.threat_memory[(h_idx - 35) % 360].item())
            right_danger = float(self.threat_memory[(h_idx + 35) % 360].item())
            if left_danger < right_danger:
                return -danger_ahead * 25.0  # Steer left away from danger
            else:
                return danger_ahead * 25.0   # Steer right away from danger
        return 0.0

    @property
    def is_escaping(self) -> bool:
        return self.escape_timer > 0

    @property
    def dopamine_efflux(self) -> float:
        return float(self.dopamine_level)

    def encode_sensory_input(self, raycast_left: float, raycast_right: float) -> torch.Tensor:
        """
        Convert left and right sensory readings (e.g. proximity/raycasts)
        into injection currents for visual neurons.

        Values can represent proximity [0.0, 1.0] or raw distances.
        Higher input stimulus leads to higher depolarizing current.
        """
        ext_current = torch.zeros((self.num_neurons,), dtype=torch.float32, device=self.device)

        # Depolarizing current scaled by sensory_gain
        i_left = float(raycast_left) * self.sensory_gain
        i_right = float(raycast_right) * self.sensory_gain

        if len(self.connectome.visual_left_indices) > 0:
            ext_current[self.connectome.visual_left_indices] = i_left
        if len(self.connectome.visual_right_indices) > 0:
            ext_current[self.connectome.visual_right_indices] = i_right

        return ext_current

    def decode_motor_output(self, spikes: torch.Tensor, return_escape: bool = False) -> Union[Tuple[float, float, int], Tuple[float, float, int, bool]]:
        """
        Extract motor signals from descending motor neuron spikes.

        Computations:
        - turn = right_motor_spikes - left_motor_spikes
        - forward = mean(all_motor_spikes)
        - total_spikes = sum(all_spikes)

        Returns
        -------
        turn : float
        forward : float
        total_spikes : int
        is_escape : bool (optional, if return_escape=True)
        """
        m_left_idx = self.connectome.motor_left_indices
        m_right_idx = self.connectome.motor_right_indices
        m_all_idx = self.connectome.motor_indices

        left_motor_spikes = float(spikes[m_left_idx].sum().item()) if len(m_left_idx) > 0 else 0.0
        right_motor_spikes = float(spikes[m_right_idx].sum().item()) if len(m_right_idx) > 0 else 0.0

        if len(m_all_idx) > 0:
            forward_signal = float(spikes[m_all_idx].mean().item())
        else:
            forward_signal = 0.0

        turn_signal = right_motor_spikes - left_motor_spikes
        total_spikes = int(spikes.sum().item())

        if return_escape:
            return turn_signal, forward_signal, total_spikes, self.is_escaping
        return turn_signal, forward_signal, total_spikes
