# 🚜 AgroNav — Navigation Simulator for Autonomous Agricultural Machinery

> **Diploma project** — 2D simulation of a beacon-based navigation system for an autonomous tractor, built on Extended Kalman Filter (EKF), UWB-beacon mesh network and boustrophedon path planning.

---

## 📌 What This Project Is About

Modern precision agriculture requires autonomous machinery that knows its exact position in a field at all times — without GPS. This project simulates such a system using a network of fixed UWB beacons and an EKF to fuse noisy odometry with beacon distance measurements.

The simulator is not a game — it is a **technical demonstration** of navigation algorithms that could run on real hardware. Every component (EKF, beacon noise model, GDOP, odometry drift) is modelled after real-world systems used in platforms like John Deere AutoTrac, CLAAS PILOT and Fendt Xaver.

---

## ✨ Key Features

### 🧭 EKF Navigation Core
- Extended Kalman Filter fuses wheel odometry with UWB beacon ranging
- Realistic odometry model: Gaussian noise + **systematic drift** proportional to distance + constant angular bias (IMU calibration error)
- **Dead Reckoning (DR) tracker** runs in parallel — same noisy odometry, zero beacon corrections — proving why EKF is necessary
- Live EKF covariance ellipse visualised on field

### 📡 Beacon Mesh Network
- 5 UWB beacons in a mesh topology (4 corners + 1 centre)
- **Adaptive noise model** per beacon: signal degrades with distance, improves when beacon has mesh neighbours (differential correction analogy)
- Mesh link visualisation: active links highlighted in real time
- **Beacon failure simulation** (toggle with `F`): beacons randomly drop out for 1–3 seconds to demonstrate EKF resilience

### 📐 GDOP — Geometric Dilution of Precision
- Standard GPS/UWB quality metric: `GDOP = √(trace((HᵀH)⁻¹))`
- **GDOP heatmap** replaces naive coverage map — shows positioning *quality*, not just beacon count
- Live GDOP value in sidebar with colour-coded rating (Excellent / Good / Fair / Poor)
- When >3 beacons are visible, system selects the **geometrically optimal triple** (minimum GDOP) — exactly how real GPS receivers select satellites

### 🗺️ Autonomous Path Planning
- **Boustrophedon (lawnmower) pattern** — standard algorithm for full-field coverage
- Path optimised using **AI Memory**: known obstacles are pre-routed around at startup
- In-flight obstacle avoidance: 3-point geometric bypass inserted into path on detection
- Emergency reverse when lidar detects imminent collision

### 🔦 Sector Lidar
- 16-ray lidar with configurable range and FOV
- Used for obstacle detection and emergency stop
- Visual ray display toggled with `L`

### 📊 Data Logging & Analysis
- CSV logger records 12 fields per frame: position, heading, speed, **EKF error**, **DR error**, EKF covariance, GDOP, visible beacons, beacon IDs, event flags
- Built-in analysis window with **7 charts**:
  - Tractor trajectory with event markers (avoidance, reverse, beacon failure)
  - **EKF vs DR error over time** — the core proof of concept
  - GDOP over time
  - EKF covariance over time
  - Error vs beacon count (box plot)
  - Error distribution histogram
  - Summary statistics table

### 📲 Telegram Alerts
- Real-time push notifications for: obstacle detected, emergency reverse, collision
- Configurable via `settings.json`

---

## 🏗️ Project Structure

```
AgroNav/
│
├── main.py                  # Main simulation loop, UI, all rendering
│
├── navigation/
│   └── ekf.py               # Extended Kalman Filter (predict + adaptive update)
│
├── environment/
│   └── beacon.py            # Beacon class: mesh, noise model, failure simulation
│
├── robot/
│   └── robot.py             # Robot kinematics + realistic odometry noise model
│
├── field.py                 # Rectangular field helper
├── utils.py                 # Config loader, math utils, Telegram, path tools
│
├── settings.example.json    # Config template (copy to settings.json)
└── ai_memory.json           # Persistent obstacle memory (auto-generated)
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/agronav.git
cd agronav

# 2. Install dependencies
pip install pygame numpy pandas matplotlib requests

# 3. Create your config
cp settings.example.json settings.json
# Edit settings.json — add your Telegram token if desired

# 4. Run
python main.py
```

**Requirements:** Python 3.9+, pygame, numpy, pandas, matplotlib, requests

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Toggle autopilot on/off |
| `E` | Path editor (draw custom route) |
| `C` | Clear AI obstacle memory |
| `BACKSPACE` | Full simulation reset |
| `S` | Start / stop CSV logging |
| `H` | Toggle GDOP heatmap |
| `L` | Toggle lidar visualisation |
| `F` | Toggle beacon failure simulation |
| Click beacon | Toggle beacon active/inactive |

---

## 🔬 How It Works — Algorithm Stack

```
┌─────────────────────────────────────────────────────────┐
│                    SENSOR LAYER                         │
│  Wheel encoders (noisy)    UWB beacons (ranging)        │
│  + systematic drift        + distance-based noise       │
│  + angular bias            + mesh correction            │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐   ┌─────────────────────────────┐
│   EKF PREDICT        │   │   EKF UPDATE                │
│   State: [x, y, θ]   │──▶│   z = measured distance     │
│   F = motion jacobian│   │   H = range jacobian        │
│   Q = process noise  │   │   R = adaptive (noise_std²) │
└──────────────────────┘   │   Best 3 beacons by GDOP    │
                           └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                  NAVIGATION LAYER                       │
│  Boustrophedon path → P-controller → v, ω commands     │
│  Obstacle scan (4 segments) → 3-point bypass insert     │
│  AI Memory → pre-route at startup                      │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Results

After a full field pass, the analysis screen shows the core result of the project:

- **EKF average error: ~5–12 px** (stable throughout the run)
- **DR average error: 30–80 px and growing** (unbounded drift without beacon correction)
- GDOP stays in the 1.5–3.5 range across most of the field with the current beacon layout
- Beacon failure mode (`F`) causes visible EKF error spikes — demonstrating system resilience

---

## 🎓 Academic Context

This project was developed as a **diploma thesis** demonstrating that a low-cost UWB beacon network combined with EKF can replace GPS for autonomous agricultural navigation in field conditions. The simulator serves as the implementation proof-of-concept.

**Core references:**
- Kalman R.E. (1960) — *A New Approach to Linear Filtering and Prediction Problems*
- Borenstein J. & Feng L. (1996) — *Measurement and Correction of Systematic Odometry Errors*
- Kaplan & Hegarty (2006) — *Understanding GPS: Principles and Applications* (GDOP chapter)
- Coulter R.C. (1992) — *Implementation of the Pure Pursuit Path Tracking Algorithm*, CMU

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.
