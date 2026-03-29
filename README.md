# 🌌 EPOCH ZERO — Orbital Defense & Fleet Surveillance

> **Physics-Informed Neural Network (PINN) Satellite Collision Prediction with Real-Time 3D WebGL Mission Control UI**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Three.js](https://img.shields.io/badge/Three.js-r170-000?logo=three.dot.js&logoColor=white)](https://threejs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00CA4E.svg)](LICENSE)

---

## 🛰️ Overview

**Epoch Zero (v5.0)** is a high-fidelity space situational awareness (SSA) platform that bridges traditional orbital mechanics with deep learning. By combining **SGP4 orbital propagation** with a **Physics-Informed Neural Network (PINN)**, the system predicts satellite conjunctions (collision risks) with accuracy that accounts for complex atmospheric drag and J2 gravitational perturbations — often missed by classical propagators alone.

The system features an immersive, **space-station-grade 3D Mission Control dashboard** built with Three.js, designed for real-time visualization of the orbital environment, fleet telemetry monitoring, and proactive threat assessment.

---

## 🔥 Key Features

### 🚀 Advanced Propagation Engine
- **Hybrid PINN Architecture**: Uses a residual-learning neural network (PyTorch) to correct SGP4 baseline trajectories, incorporating physics-based constraints.
- **Precision Correction**: Accounts for atmospheric drag (using space weather F10.7 indices) and Earth's oblateness (J2).
- **Batch Processing**: Simultaneously propagates and monitors entire satellite constellations or debris clouds.

### 🗺️ Inter-Orbital Visualization (WebGL)
- **High-Performance 3D Globe**: Rendered with Three.js (r170) featuring realistic Earth textures, atmospheric glow, and star fields.
- **Dynamic Orbital Tails**: Satellites leave "dynamic tails" (streaming trajectory windows) rather than static loops for more realistic motion.
- **Collision Heatmaps**: Real-time coloring of conjunction lines (Kill Zones) based on proximity (Red < 100km, Orange < 500km).
- **Cinematic Controls**: Auto-rotate, camera tracking, and "Warp Speed" (up to 10x real-time) simulation playback.

### 🛡️ Mission Operations
- **Live TLE Fetch**: Automated batch fetching of Two-Line Elements (TLE) from CelesTrak.
- **Threat Detection**: Automatically identifies the top 20 most dangerous conjunctions from a selected fleet.
- **Fleet Telemetry**: Live readout of altitude, orbital velocity, and PINN-corrected position vectors.
- **Manual Target Lock**: Lock onto any NORAD object by ID to add it to the active tracking pool.

---

## 🛠️ Tech Stack

| Domain | Technology |
|---|---|
| **Core Language** | Python 3.12+ (Backend) \| Javascript/ESM (Frontend) |
| **Backend API** | Flask (High-performance JSON endpoints) |
| **Neural Engine** | PyTorch (Residual PINN v3.3) |
| **Physics** | SGP4 (python-sgp4) with J2 + Atmosphere Drag Logic |
| **3D Rendering** | Three.js (WebGL 2.0) with custom GLSL Shaders |
| **Styling** | Modern CSS (Glassmorphism, Dark Mode, Space Station Layout) |
| **Data Source** | CelesTrak (Live TLE Feed) |

---

## 📂 Project Architecture

```text
Project-Epoch-Zero/
├── server.py           # Main Entry Point — Flask API Server
├── src/
│   ├── deployer.py     # OrbitDeployer — Hybrid Physics/PINN Pipeline
│   ├── model.py        # Gated/Residual PINN Neural Network Architecture
│   ├── tle_fetcher.py  # CelesTrak API Integration
│   ├── train.py        # Model Training Pipeline
│   └── utils.py        # Coordinate conversions (ECI/LLA/ECEF)
├── static/             # WebGL Frontend
│   ├── index.html      # Mission Control UI Frame
│   ├── app.js          # Three.js Visualization Engine (50KB+)
│   └── style.css       # Space Station Design System
├── models/             # Neural Network Weights
│   ├── pinn_model.pth  # Trained PINN State Dict
│   └── scaler_*.pkl    # Feature Normalization Layers
├── notebooks/          # R&D and Model Validation
├── tests/              # Unit & Integration Testing
└── pyproject.toml      # Dependency Management
```

---

## 🚦 Quick Start

### 1. Prerequisites
- **Python 3.12+**
- (Optional but recommended) **uv** or **conda** for environment management.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Sibikrish3000/Project-Epoch-Zero.git
cd Project-Epoch-Zero

# Install dependencies (using uv)
uv sync

# OR using pip
pip install -r requirements.txt
```

### 3. Launch System
```bash
# Start the Flask mission control server
python server.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser (Chrome/Edge/Safari recommended for WebGL performance).

---

## 🕹️ Mission Protocol

1.  **Initialize Fleet**: Select a satellite group (e.g., *ISS*, *Starlink*) or debris group (e.g., *Iridium-33 fragments*) and click **Load**.
2.  **Sync Data**: Click **Fetch Live TLEs** to populate the tracking engine with real-time orbital elements.
3.  **Define Epoch**: Set the assessment **Target Date** and **Propagation Window**.
4.  **Execute Scan**: Hit **⚡ EXECUTE BATCH SCAN**. The backend will compute PINN-corrected paths for all objects.
5.  **Visualize**: Use the **Timeline Scrubber** or **Play** button to watch the satellites orbit. Red conjunction lines indicate high-risk encounters.
6.  **Analyze**: Monitor the **Collision Risk Matrix** and **Top Threats** panels for miss-distance data and confidence levels.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

*Disclaimer: This system is for research and situational awareness simulation purposes only.*
