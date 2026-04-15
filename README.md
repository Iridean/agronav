# 🚜 AgroNav: Autonomous Tractor Navigation Simulator

> **My Diploma Project (@ MTUCI)**. A 2D simulation of a GPS-less navigation system for agricultural machinery, utilizing an Extended Kalman Filter (EKF), UWB-beacon mesh network, and boustrophedon path planning.

Precision agriculture requires machinery to know its exact position without relying solely on GPS. I built this simulator as a **technical proof-of-concept** to demonstrate how sensor fusion can solve this. It models real-world physics and algorithms similar to those used in commercial platforms like John Deere AutoTrac or CLAAS PILOT.

### ✨ Key Features

- **Sensor Fusion (EKF):** An Extended Kalman Filter fuses noisy wheel odometry (featuring systematic drift and IMU bias) with UWB beacon ranging. It runs alongside a Dead Reckoning (DR) baseline to visually prove EKF's necessity.
- **UWB Mesh & GDOP:** Simulates a 5-beacon network with adaptive noise and failure simulation. The system dynamically selects the optimal beacon triple based on **Geometric Dilution of Precision (GDOP)** heatmaps.
- **Smart Path Planning:** Implements the Boustrophedon (lawnmower) algorithm. Features a 16-ray sector Lidar for real-time obstacle avoidance and an **AI Memory** system to pre-route around known field obstacles.
- **Analytics & Alerts:** Includes a CSV logger tracking 12 parameters per frame. The built-in analysis tool generates 7 Matplotlib charts (EKF vs DR error, covariance, GDOP). Plus, real-time Telegram push notifications for emergency stops.

### 🛠 Tech Stack
**Python 3.9+** | `NumPy` (Matrix math/EKF) | `Pandas` & `Matplotlib` (Data analysis) | `Pygame` (Rendering) | `Requests` (Alerts)

### 🚀 Quick Start

```bash
git clone https://github.com/Iridean/agronav.git
cd agronav

# Install dependencies
pip install pygame numpy pandas matplotlib requests

# Set up config (add Telegram token if needed)
cp settings.example.json settings.json

python main.py
🎮 Controls:
SPACE: Autopilot | E: Path Editor | C: Clear AI Memory | S: Toggle CSV Logging
H: GDOP Heatmap | L: Lidar View | F: Simulate Beacon Failure | Click: Toggle Beacons
🔬 Architecture & Results
The core achievement of this thesis is the algorithm stack:
Odometry + UWB Ranging ➔ EKF Predict/Update Loop ➔ P-Controller ➔ Obstacle Avoidance
After running a full field pass, the built-in analytics clearly validate the system:
Dead Reckoning (DR) error: Grows unboundedly to 30–80+ px.
EKF average error: Stays stable at ~5–12 px, effectively proving that a low-cost UWB network combined with EKF can successfully replace GPS in autonomous field conditions.
