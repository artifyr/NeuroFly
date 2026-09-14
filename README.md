# NeuroFly: Connectome-Driven Spiking Neural Network (SNN) & Minecraft Embodiment

[![GitHub repository](https://img.shields.io/badge/GitHub-artifyr%2FNeuroFly-blue?logo=github)](https://github.com/artifyr/NeuroFly)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm%20%7C%20CUDA-orange.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Dataset-Git%20LFS-red.svg)](https://git-lfs.github.com/)

A biophysical, full-scale simulation of the adult fruit fly (*Drosophila melanogaster*) connectome (**138,584 neurons**, **~15 million synapses**) from the Janelia Full Adult FlyBrain (FAFB) dataset. Hardware-accelerated via PyTorch on AMD ROCm / NVIDIA CUDA GPU, streaming real-time neural spikes, neuromodulator dynamics, and descending motor efferents directly to an in-game **Minecraft Fly/Bee Agent**.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────┐
│               Minecraft Game / World                   │
│   (Player, Torches, Lava, Food, Bee Entity via MCP)    │
└────────────────────────▲───────────────────────────────┘
                         │ Fast MCP HTTP (:8080)
┌────────────────────────▼───────────────────────────────┐
│              NeuroFly Brain Engine                     │
│             (run_minecraft_brain.py)                   │
│  - 138,584 LIF Neuron Biological Connectome on GPU     │
│  - Neuro-Ethological Circuit Behaviors & SNN Motifs    │
│  - Real-time 20Hz Biological Saccadic Flight Loop      │
│  - In-Game Floating Actionbar Biometric Telemetry HUD   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Key Neuro-Ethological Features

### 🧠 Biophysically Grounded 138k LIF SNN
- **Leaky Integrate-and-Fire (LIF)** network with membrane voltage decay ($\tau = 20\text{ ms}$), resting potential ($V_{rest} = -70\text{ mV}$), threshold detection ($V_{thresh} = -50\text{ mV}$), and refractory clamping.
- **Sub-3ms GPU Step Latency** on AMD Radeon ROCm / NVIDIA CUDA.

### 🐝 Biological Minecraft Fly / Bee Behaviors
1. **60° Posterior Blind Spot (Physical Compound Eye Geometry)**:
   - Bilateral eyes wrap around the head with a $\approx 60^\circ$ posterior optical gap ($|\theta_{rel}| > 150^\circ$).
   - Giant Fiber looming takeoff reflexes only fire when approaching from front/sides; players can sneak up directly behind the fly undetected!
2. **Johnston's Organ (Acoustic Vibration Transduced as Touch & In-Place Annoyance)**:
   - Footsteps, tool sounds, or physical closeness within $1.5\text{ m}$ stimulate antennal chordotonal mechanoreceptors.
   - Triggers an annoyed vertical flinch hop buzzing up and down in place with a smoke particle burst, then automatically returns to its previous task.
3. **Metabolic Energy Pool & Neuropeptide F (NPF) Hunger Dominance**:
   - Continuous flight depletes glycogen reserves. When energy drops below $50\%$, NPF sensitizes projection neurons up to $3.5\times$, prioritizing sugar/melon/honey odor tracking over wandering.
   - Feeding restores energy and heals in-game HP.
4. **Rain Avoidance & Overhead Shelter Seeking**:
   - Live telemetry monitors Minecraft weather. Raindrops are lethal impacts; the fly drops to low vegetation, scans for overhead cover (tree leaves, cave roofs, ceilings) in an 8-block radius, and perches sheltered while grooming until clear skies return.
5. **Circadian Sleep & Stereotyped 3-Stage Grooming**:
   - Perches quietly during Minecraft night with sensory gating.
   - Executes stereotypic Antennae $\rightarrow$ Eyes $\rightarrow$ Wings & Legs grooming sequences.
6. **Phototaxis & Thermal Nociception**:
   - Smooth orbital banking around torches/lanterns; sharp repulsive thermal avoidance away from lava, fire, and campfires.
7. **Two-Line In-Game Actionbar HUD**:
   - Real-time zero-clipping HUD alternating between identity/state (`[FAFB Bee] HP: 20 [♥♥♥♥♥] [HUNGRY]`) and neural/environmental metrics (`Spikes: 1,420 | Nrg: 42% | PAM: 0.85 | Rear: [BLIND] | Cov: Sheltered`).

---

## 3. Quick Start Guide

### Prerequisites
- Python 3.10+ with PyTorch (ROCm, CUDA, or CPU fallback)
- Git & [Git LFS](https://git-lfs.github.com/)
- Minecraft Java Edition (Supports both the Official Minecraft Launcher and 3rd-Party Launchers) with Fabric Loader + MCP Server Mod

---

### Step 1: Clone Repository & LFS Assets
```bash
git clone https://github.com/artifyr/NeuroFly.git
cd NeuroFly
git lfs pull
```
*(Git LFS fetches the Princeton synapse dataset and connectome annotations).*

### Step 2: Install Python Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Connecting to Minecraft (Embodiment Setup)

NeuroFly controls in-game entities using high-speed JSON-RPC over the Model Context Protocol (MCP) mod server running on port `8080`.

### Option A: Official Minecraft Launcher
1. **Install Fabric Loader**:
   - Download and run the Fabric installer from [fabricmc.net](https://fabricmc.net/) for Minecraft Java Edition.
2. **Install Fabric API & MCP Server Mod**:
   - Place `fabric-api.jar` and the MCP server mod (e.g., `mcp-server-mod-*.jar`) into your official `.minecraft/mods` directory:
     - **Windows**: `%appdata%\.minecraft\mods\`
     - **macOS**: `~/Library/Application Support/minecraft/mods/`
     - **Linux**: `~/.minecraft/mods/`
3. **Launch Minecraft**:
   - Start the game using the Fabric profile in the official launcher.
   - Enter a Singleplayer world (or multiplayer LAN server with cheats/operator enabled).
   - Ensure the MCP server is listening (default port `8080`).

### Option B: 3rd-Party Launchers
1. Create or edit an existing Fabric instance matching your Minecraft version.
2. Under your instance's **Mods** tab, click **Add / Open Mods Folder** and add:
   - `fabric-api`
   - `mcp-server-mod`
3. Launch the instance and join your world.

### Step 3: Launch the Biological Brain
Once inside your Minecraft world, run the biological SNN flight controller:
```bash
python run_minecraft_brain.py --entity bee
```
- **Autonomous Initialization**: The controller automatically locates or spawns your biological agent, anchors the 20Hz biological simulation loop, and begins streaming live spikes and sensory-motor commands.
- **In-Game Actionbar HUD**: Watch the in-game actionbar display live biometrics (`[FAFB Bee] HP: 20`, current neural firing rates, hunger states, and environmental cover).

*(Optional flags: `--rain` to test rain shelter avoidance, `--walk` for ground walking mode).*

---

## 5. Repository Structure

```
NeuroFly/
├── connectome_loader.py      # FAFB 138k connectome parser & adjacency matrix builder
├── snn_engine.py             # PyTorch GPU Spiking Neural Network (LIF + STDP + Neuromodulators)
├── run_minecraft_brain.py    # Main embodiment loop: MCP Bridge & 20Hz biological flight physics
├── minecraft_fly_bridge.py   # High-speed JSON-RPC bridge with in-memory JVM hot-reloading
├── data/                     # FAFB connectome metadata & Princeton synapse dataset (Git LFS)
│   ├── connections_princeton.csv  (Tracked by Git LFS)
│   ├── classification.csv
│   ├── neurons.csv
│   └── visual_neuron_types.csv
├── tests/                    # Behavioral & neural unit tests
└── scratch/                  # Behavior verification & hot-reload agent tools
```

---

## 6. Verification & Tests

Run the full automated behavioral test suite:
```bash
python tests/test_behavioral_suite.py
```

All 5 core neuro-ethological suites verified:
- [x] 60° Posterior Blind Spot
- [x] Johnston's Organ Mechanosensory Touch & In-Place Annoyance
- [x] Metabolic Energy Pool & Hunger Threshold ($< 50\%$)
- [x] Rain Avoidance & Overhead Shelter Seeking
- [x] Two-Line Actionbar HUD Length & Formatting

---

## 7. Scientific References & Credits
- **Connectome Dataset**: Janelia Research Campus & Princeton University (*Full Adult FlyBrain - FAFB*), Schlegel et al., Nature 2024 / FlyWire Consortium.
- **Olfactory & Feeding Circuits**: Cognigni et al., *Neuropeptide F Signaling in Drosophila Feeding Motivation*.
- **Giant Fiber Visual Escape**: Card & Dickinson, *Visually mediated motor planning in the escape response of Drosophila*.
