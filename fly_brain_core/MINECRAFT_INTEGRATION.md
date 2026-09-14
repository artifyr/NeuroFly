# Drosophila Fly Brain (SNN) Integration Guide for Minecraft & 3D WebApp

This document provides complete instructions, protocol specifications, data structures, and ready-to-use bridge scripts for connecting your **Minecraft Fly/Bee entity** to the live **FlyBrain Spiking Neural Network (SNN)** simulation server and visualizing the fly's brain in real time directly in the **3D Drosophila WebApp Visualizer**.

---

## 1. Overview & Architecture

The **FlyBrain SNN Server** runs a biological LIF (Leaky Integrate-and-Fire) simulation of the *Drosophila melanogaster* connectome (138,584 neurons, ~15 million synapses), hardware-accelerated via PyTorch & ROCm/CUDA on GPU.

It exposes a bi-directional **WebSocket API** on port `8765`:

```
┌────────────────────────────────────────────────────────┐
│               Minecraft Game / World                   │
│   (Player, Torches, Lava, Food, Bee Entity via MCP)    │
└────────────────────────▲───────────────────────────────┘
                         │ MCP HTTP (:8080)
┌────────────────────────▼───────────────────────────────┐
│        FlyAI Minecraft Brain & Flight Controller       │
│               (run_minecraft_brain.py)                 │
│  - 138,584 LIF Neuron Biological Connectome on ROCm   │
│  - All Biological Behaviors (Feeding, Sleep, Grooming) │
│  - Embedded WebSocket Telemetry Server (:8765)         │
└────────────────────────▲───────────────────────────────┘
                         │ WebSocket Broadcast (:8765)
                         │ (Spikes + Motor + Dopamine + Minecraft Agent HUD)
┌────────────────────────▼───────────────────────────────┐
│         3D Drosophila WebApp Visualizer                │
│               (http://localhost:5173)                  │
│       - 138,584 Neuron Somas & Morphological Skeletons │
│       - Real-time Minecraft Fly HUD (Pos, HP, State)   │
│       - Dynamic Sensory Injections & Circuit Waves     │
└────────────────────────────────────────────────────────┘
```

- **Efferent Motor Outputs (Brain -> Minecraft)**: Live flight thrust (`forward`), yaw steering torque (`turn`), Giant Fiber escape jump trigger (`is_escaping`), and dopamine neuromodulation.
- **Afferent Sensory Inputs (Minecraft -> Brain)**: Sugar/sweet reward, looming visual threat, nociceptive pain shock, or food odor plume injected directly into the fly's brain during gameplay.
- **In-Game Entity Telemetry (Minecraft -> Brain -> WebApp)**: Real-time 3D coordinates, pitch/yaw, health points, and biological behavioral state displayed inside the WebApp's **Minecraft Fly HUD**.


---

## 2. WebSocket Communication Protocol

### Connection Endpoint
```
ws://localhost:8765
```
*(If running on a different machine or LAN, replace `localhost` with the server's IP address).*

---

### A. Receiving Brain & Agent Telemetry (Server -> WebApp & Minecraft)

Every simulation step (~25–40 FPS / ~25ms interval), the SNN server broadcasts a JSON payload containing neural firing data, motor efferents, and connected Minecraft agent state:

#### JSON Telemetry Format:
```json
{
  "type": "telemetry",
  "step": 4820,
  "count": 142,
  "forward": 0.78,
  "turn": -0.35,
  "is_escaping": false,
  "dopamine": 0.42,
  "appetitive": 0.35,
  "escape_timer": 0,
  "latency_ms": 2.15,
  "spikes": [142, 891, 2045, 12055, 34091],
  "agent": {
    "connected": true,
    "source": "Minecraft",
    "entity": "bee",
    "pos": [-38.2, -58.0, 175.4],
    "yaw": 45.0,
    "pitch": -2.1,
    "hp": 20.0,
    "max_hp": 20.0,
    "state": "FEEDING (Flower) [+HP]",
    "speed": 2.4,
    "nearest_light": {"dist": 2.4, "name": "Torch"},
    "nearest_hazard": {"dist": 8.0, "name": "Campfire"},
    "nearest_food": {"dist": 1.2, "name": "Flower (Sugar)"},
    "player_dist": 4.5,
    "is_night": false,
    "last_seen": 1726284900.5
  }
}
```

#### Field Specifications:

| Field | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `type` | `string` | `"telemetry"` | Message identifier header. |
| `step` | `number` | Counter | Current SNN simulation step index. |
| `count` | `number` | Count | Total active spiking neurons in this timestep. |
| `forward` | `number` | `0.0` to `1.0` | **Forward Flight Thrust**: Derived from Descending Motor Neuron firing rates. Map to entity forward velocity. |
| `turn` | `number` | `-1.0` to `+1.0` | **Yaw Steering Torque**: Negative = Left turn torque, Positive = Right turn torque. Map to entity yaw rotation. |
| `is_escaping` | `boolean` | `true`/`false` | **Giant Fiber Escape Reflex**: Instantaneous jump/takeoff trigger activated by visual looming threat or shock. |
| `dopamine` | `number` | `0.0` to `1.0` | **PAM/PPL1 Dopamine Level**: Real-time neuromodulator state for reinforcement learning / mood. |
| `spikes` | `array[int]` | Neuron IDs | Array of exact neuron IDs firing in this timestep (renders glowing soma flashes in WebApp). |
| `agent` | `object` | Metadata | Live in-game Minecraft entity state (coordinates, HP, behavior state label, surroundings radar). |

---

### B. Streaming In-Game Agent Telemetry (Minecraft -> Server -> WebApp)

The Minecraft bridge streams its in-game entity pose and environmental perceptions to the SNN server:

#### JSON Agent Telemetry Payload:
```json
{
  "type": "agent_telemetry",
  "source": "Minecraft",
  "entity": "bee",
  "pos": [-38.2, -58.0, 175.4],
  "yaw": 45.0,
  "pitch": -2.1,
  "hp": 20.0,
  "max_hp": 20.0,
  "state": "PHOTOTAXIS (Orbiting Torch)",
  "speed": 2.4,
  "nearest_light": {"dist": 2.4, "name": "Torch"},
  "nearest_hazard": {"dist": 8.0, "name": "Campfire"},
  "nearest_food": {"dist": 1.2, "name": "Flower"},
  "player_dist": 4.5,
  "is_night": false
}
```

---

### C. Sending Sensory Stimuli (Minecraft / WebApp -> Brain)

Stimuli injected during gameplay immediately excite biological neural circuits:

#### JSON Stimulus Payload:
```json
{
  "action": "<STIMULUS_ACTION>",
  "intensity": 1.0
}
```

#### Available Stimuli Actions:

| Action Command | Intensity | Brain Target Circuit | Biological Effect on Fly |
| :--- | :--- | :--- | :--- |
| `"inject_sweet"` | `1.0` | Sugar Gustatory Receptors $\rightarrow$ PAM Dopamine & MB | Triggers feeding behavior, reward learning, and positive dopamine efflux. |
| `"inject_looming"`| `1.0` | Lobula Plate Visual Neurons $\rightarrow$ Giant Fiber (GF) | Triggers **Giant Fiber Escape Reflex** (`is_escaping = true`), high forward thrust, and rapid flight takeoff. |
| `"inject_pain"` | `1.0` | Nociceptive Sensory Circuit $\rightarrow$ PPL1 Dopamine | Triggers aversive avoidance, sharp steering turns, and dopamine drop. |
| `"inject_odor"` | `1.0` | Antennal Lobe Glomeruli $\rightarrow$ Kenyon Cells | Activates olfactory search behavior and sparse associative memory encoding. |
| `"set_phase"` | `is_night: true` | Clock Neurons (s-LNvs, l-LNvs, DN1) | Enters Circadian sleep quiescence on perch. |
| `"reset"` | `1.0` | Whole Brain Reset | Restores membrane potentials, active spikes, and neuromodulators to baseline. |

---

## 3. Ready-to-Use Scripts

### Script 1: Full Minecraft Fly Bridge (`minecraft_fly_bridge.py`)
Connects directly to `ws://localhost:8765` and Minecraft via MCP (`http://localhost:8080/mcp`):

```bash
# Live mode with Minecraft MCP server running:
python minecraft_fly_bridge.py --server ws://localhost:8765 --entity bee

# Instant test mode (simulates Minecraft world and feeds live telemetry to WebApp):
python minecraft_fly_bridge.py --mock
```

### Script 2: Autonomous GPU Flight Controller (`run_minecraft_brain.py`)
Runs the full 138k SNN locally on GPU (AMD Radeon RX 9070 XT ROCm), controls the Minecraft entity, and streams live telemetry to the WebApp:

```bash
python run_minecraft_brain.py --entity bee
```

---

## 4. How to Visualize the Minecraft Fly in the WebApp

1. **Start the 3D Web Visualizer**:
   ```bash
   cd web
   npm run dev
   ```
2. **Open the 3D Drosophila WebApp**:
   Navigate to `http://localhost:5173` in your browser.
3. **Run the Minecraft Brain**:
   ```bash
   python run_minecraft_brain.py --entity bee
   ```
4. **Observe Live In-Game Synchronization**:
   - The **Minecraft Fly HUD** in the WebApp top-left activates with a glowing green badge (`[MINECRAFT FLY: ACTIVE]`).
   - Real-time coordinates `(X, Y, Z)`, health bar hearts (`♥♥♥♥♥♥♥♥♥♥`), and behavior states (`[FEEDING]`, `[PHOTOTAXIS]`, `[ESCAPE]`) update dynamically.
   - Click **"Loom Escape"** or **"Feed Sugar"** in the WebApp to trigger instantaneous spike bursts across the 138k connectome and watch the Minecraft fly react in the game world!
