import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from snn_engine import FlyBrainSNN
from connectome_loader import load_connectome_from_csv

device = "cuda:0" if torch.cuda.is_available() else "cpu"
connectome = load_connectome_from_csv("data/connections_princeton.csv", "data/classification.csv", device=device)
snn = FlyBrainSNN(connectome)

print("--- SLEEP WITH BIOLOGICAL GATING & HYPERPOLARIZATION ---")
snn.set_circadian_phase(is_night=True)

# Test with sleep modulation
# 1. Sensory gating: no optic flow in sleep
# 2. Spontaneous noise: dropped by ~95%
# 3. Synaptic gain / threshold hyperpolarized
orig_thresh = snn.v_thresh
orig_gain = snn.synaptic_gain

snn.v_thresh = orig_thresh + 8.0  # Hyperpolarized threshold during sleep
snn.synaptic_gain = orig_gain * 0.35  # GABAergic sleep tone

for t in range(10):
    # Only sparse basal resting fluctuations + circadian clock pacemaker
    spontaneous = (torch.rand(snn.num_neurons, device=device) < 0.00015).float() * 8.0
    circ_idx = getattr(connectome, "circadian_indices", None)
    if circ_idx is not None and len(circ_idx) > 0:
        spontaneous[circ_idx] += 8.0  # Circadian clock neurons remain gently rhythmic

    spikes = snn.step(spontaneous)
    count = int(spikes.sum().item())
    print(f"Modulated Sleep Tick {t}: {count:,} spikes")

# Arousal test: Looming threat shocks the fly awake!
print("\n--- THREAT AROUSAL (WAKE UP!) ---")
snn.v_thresh = orig_thresh
snn.synaptic_gain = orig_gain
pain_cur = snn.inject_pain(intensity=1.0)
spikes = snn.step(pain_cur)
print(f"Arousal Escape Spike Burst: {int(spikes.sum().item()):,} spikes!")
