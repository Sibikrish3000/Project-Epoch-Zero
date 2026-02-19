# EPOCH ZERO — Orbital Defense System

> **PINN-Enhanced Satellite Collision Prediction with Real-Time 3D Mission Control UI**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Dash](https://img.shields.io/badge/Dash-2.15+-00d4ff?logo=plotly&logoColor=white)](https://dash.plotly.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Epoch Zero is a **space-station-grade conjunction assessment system** that combines SGP4 orbital propagation with a Physics-Informed Neural Network (PINN) to predict satellite collision risks with higher accuracy than traditional methods alone.

The system features a **real-time 3D mission control dashboard** built with Dash and Plotly, designed for space situational awareness operators to visualize orbital encounters, assess collision probabilities, and monitor satellite telemetry — all in a single interface.

---

## UI Architecture

The interface follows a **3-panel mission control layout** modeled after real-world space operations centers:

```
┌──────────────────────────────────────────────────────────────────────┐
│  HEADER BAR — System status, PINN status, UTC clock                 │
├──────────┬─────────────────────────────────┬─────────────────────────┤
│          │                                 │                         │
│  LEFT    │     CENTER VIEWPORT             │    RIGHT                │
│  SIDEBAR │                                 │    SIDEBAR              │
│          │     Real-Time 3D Globe          │                         │
│  Mission │     + Animated Satellites       │    Telemetry            │
│  Control │     + Orbit Trajectories        │    + Conjunction Data   │
│  Panel   │     + Kill Zone / LOS           │    + Satellite Stats    │
│          │     + Star Field                │    + PINN Corrections   │
│          │     + Earth Grid                │    + Event Log          │
│          │                                 │                         │
├──────────┴─────────────────────────────────┴─────────────────────────┤
│  BOTTOM BAR — Model info, engine status, propagator status          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## UI Components — Detailed Breakdown

### Header Bar
| Element | Description |
|---|---|
| **Brand Logo** | "EPOCH ZERO" with satellite icon |
| **System Status** | Green pulsing dot — SYS NOMINAL / OFFLINE |
| **PINN Status** | Blue dot — indicates if GatedPINN model loaded successfully |
| **UTC Clock** | Live-updating UTC timestamp, refreshes every second |
| **Subtitle** | "ORBITAL DEFENSE SYSTEM v2.0 \| MISSION CONTROL" |

### Left Sidebar — Mission Control Panel

#### 1. Object Selection
- **Primary Object** dropdown — select the satellite you are tracking (e.g., STARLINK-3321)
- **Secondary Object** dropdown — select the threat object (e.g., COSMOS-1408 Debris)
- Pre-loaded satellite database with 4 objects (expandable):
  - `STARLINK-3321` [50803] — LEO Payload
  - `COSMOS-1408 DEB` [99999] — Debris from ASAT test
  - `ISS (ZARYA)` [25544] — International Space Station
  - `STARLINK-2305` [48274] — LEO Payload

#### 2. Prediction Window
- **Target Date** — Date picker for the conjunction assessment epoch
- **Target Time (UTC)** — Manual time input in `HH:MM:SS` format
- **Propagation Window Slider** — 30 minutes to 3 hours, controls the trajectory arc length

#### 3. Animation Controls
- **Playback Speed Slider** — 0.5x to 5x speed
- **RUN** button — Start playback animation
- **PAUSE** button — Toggle pause/resume
- **STEP** button — Advance one frame

#### 4. Visual Layers (Toggle Switches)
| Layer | Description |
|---|---|
| **Orbit Trajectories** | Satellite orbital paths — bright trail (past) + dotted (future) |
| **Kill Zone / LOS** | Line-of-sight between objects — color changes by distance (red < 1000km, orange < 3000km) |
| **Atmosphere** | Semi-transparent blue glow around Earth |
| **Star Field** | 800 randomized background stars at varying sizes |
| **Grid Lines** | Latitude/longitude grid on Earth's surface |
| **Velocity Vectors** | Green/red arrows showing velocity direction and magnitude |

#### 5. Scan Button
- **"INITIATE CONJUNCTION SCAN"** — Triggers PINN model inference, computes trajectories, and starts the animation

### Center Viewport — 3D Orbital Visualization

The centerpiece of the UI — a fully interactive Plotly 3D scene:

- **Earth** — 50-point resolution sphere with ocean-depth blue colorscale and realistic lighting
- **Atmosphere** — 1.5% larger translucent shell with blue glow effect
- **Satellites** — Diamond marker (primary, cyan) and circle marker (secondary, orange), labeled with names
- **Orbit Trails** — Animated trajectory lines that grow frame-by-frame:
  - Bright solid line = path already traveled
  - Dim dotted line = predicted future path
- **Kill Zone Line** — Dotted line connecting both objects, color shifts based on proximity
- **Auto-Rotating Camera** — Slowly orbits the scene during animation for cinematic effect
- **Frame Counter** — Shows current frame / total frames (e.g., `47/200`)
- **Turntable Drag** — Click and drag to rotate manually
- **Scroll Zoom** — Mouse wheel to zoom in/out

### Right Sidebar — Telemetry & Assessment

#### 1. Conjunction Assessment
| Metric | Description |
|---|---|
| **Miss Distance** | PINN-corrected closest approach distance in km |
| **Collision Probability** | Estimated collision probability (from `> 1e-2` to `< 1e-7`) |
| **Time to TCA** | Target time of closest approach |
| **Threat Level Badge** | Color-coded pulsing indicator with 5 levels: |

**Threat Levels:**
| Level | Distance | Color | Animation |
|---|---|---|---|
| **STANDBY** | No data | Gray | None |
| **LOW** | > 2000 km | Green | None |
| **WARNING** | 500–2000 km | Orange | Slow pulse |
| **HIGH** | 100–500 km | Red | Fast pulse |
| **CRITICAL** | < 100 km | Deep red | Rapid pulse + glow |

#### 2. Primary Telemetry (SAT-A)
- **Position (PINN)** — Corrected [x, y, z] coordinates in km
- **Velocity** — Orbital velocity magnitude in km/s
- **Altitude** — Height above Earth's surface in km

#### 3. Secondary Telemetry (SAT-B)
- Same metrics as primary, displayed in orange accent

#### 4. PINN Correction Panel
| Metric | Description |
|---|---|
| **SGP4 → PINN Δr (A)** | Position correction magnitude for primary object |
| **SGP4 → PINN Δr (B)** | Position correction magnitude for secondary object |
| **Model Confidence** | HIGH if both corrections < 5 km, else MODERATE |

#### 5. Event Log
- Scrollable log with color-coded tags:
  - `SYS` (blue) — System events
  - `CMD` (purple) — User commands
  - `OK` (green) — Successful operations
  - `ERR` (red) — Errors and failures
- Auto-trims to last 20 entries

### Bottom Status Bar
- Model identifier: `GatedPINN v3.1.1`
- Engine: `SGP4 + J2/Drag`
- Propagator status: `ACTIVE` (green)

---

## Design System

### Color Palette
| Token | Hex | Usage |
|---|---|---|
| Primary | `#00d4ff` | Cyan — primary satellite, UI accents, buttons |
| Secondary | `#ff6b35` | Orange — threat object, warnings |
| Success | `#00ff88` | Green — nominal status, low threat |
| Danger | `#ff0040` | Red — critical alerts, errors |
| Purple | `#a855f7` | Model confidence indicators |
| Background | `#05080f` | Deep space black |
| Surface | `#0a1628` | Card/panel backgrounds |
| Border | `#0d1f30` | Subtle separators |

### Typography
| Font | Usage |
|---|---|
| **Orbitron** | Section headers, threat badges, brand text |
| **Rajdhani** | Body text, labels, buttons |
| **Share Tech Mono** | Telemetry values, timestamps, coordinates, log entries |

### Visual Effects
- **Pulsing status dots** — Animated opacity cycle for system/model status
- **Threat badge animations** — Warning/critical states pulse at increasing frequencies
- **Glassmorphism panels** — Semi-transparent backgrounds with backdrop blur
- **Gradient buttons** — Cyan-to-blue gradient on primary actions with hover glow
- **Scan line effect** — CSS animation overlay during active scans

---

## How the Real-Time Animation Works

```
1. User clicks "INITIATE CONJUNCTION SCAN"
     │
2. Backend runs SGP4 propagation for both objects
   └─ Generates 200-step trajectory arrays (T-50min to T+50min)
     │
3. PINN model corrects final positions
   └─ GatedPINN applies learned corrections to SGP4 output
     │
4. All data stored in dcc.Store (client-side JSON)
     │
5. dcc.Interval fires every N ms (controlled by speed slider)
   └─ Each tick advances frame counter by 1
     │
6. render_frame() callback rebuilds the 3D figure:
   ├─ Draws orbit trail up to current frame (bright)
   ├─ Draws remaining trail (dim, dotted)
   ├─ Places satellite markers at current frame position
   ├─ Updates camera angle (slow auto-rotation)
   ├─ Computes real-time distance between objects
   └─ Updates all telemetry readouts
     │
7. Animation loops back to frame 0 when complete
```

The `uirevision='constant'` parameter prevents Plotly from resetting the camera on each frame update, enabling smooth continuous animation.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Dash 2.15+, Plotly.js, Dash Bootstrap Components |
| **3D Rendering** | Plotly Scatter3d + Surface (WebGL) |
| **ML Model** | PyTorch GatedPINN (Physics-Informed Neural Network) |
| **Propagator** | SGP4 (python-sgp4) with J2 + Drag perturbations |
| **Styling** | Custom CSS, Google Fonts (Orbitron, Rajdhani, Share Tech Mono) |
| **Icons** | dash-iconify (Material Design Icons) |

---

## Project Structure

```
satellite-collision-prediction/
├── src/
│   ├── app.py          # Main Dash application (UI + callbacks)
│   ├── deployer.py     # OrbitDeployer — SGP4 + PINN inference pipeline
│   ├── model.py        # GatedPINN neural network architecture
│   ├── train.py        # Training script
│   └── utils.py        # Constants and seed utilities
├── assets/
│   └── style.css       # Space station mission control theme
├── models/
│   ├── pinn_model.pth  # Trained PINN weights
│   ├── scaler_X.pkl    # Input feature scaler
│   └── scaler_Y.pkl    # Output target scaler
├── tests/
│   ├── test_app_load.py
│   └── test_model_load.py
├── notebooks/
│   └── notebook_pinn_v3.1.1.ipynb  # Training notebook
├── pyproject.toml      # Dependencies (uv/pip)
└── README.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Puvinthar/satellite-collision-prediction.git
cd satellite-collision-prediction

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r pyproject.toml

# Run the application
python -m src.app
```

Open **http://127.0.0.1:8051** in your browser.

### Usage
1. Select **Primary** and **Secondary** objects from the dropdowns
2. Set the **Target Date** and **Time** for conjunction assessment
3. Adjust the **Propagation Window** (30min–3hrs)
4. Click **INITIATE CONJUNCTION SCAN**
5. Watch the real-time 3D animation — satellites orbit Earth with live telemetry
6. Use **Play/Pause/Step** to control animation, **Speed slider** to adjust rate
7. Toggle visual layers on/off as needed
8. Monitor the **Threat Level**, **Miss Distance**, and **PINN Corrections** in the right panel

---

## License

MIT — See [LICENSE](LICENSE)
