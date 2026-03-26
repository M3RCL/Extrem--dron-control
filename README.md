# Drone Obstacle Avoidance — SAC with Depth Perception

Reinforcement learning agent that pilots a quadrotor through obstacle clusters to reach a goal, trained entirely in [CosysAirSim](https://github.com/Cosys-Lab/Cosys-AirSim). The policy learns obstacle avoidance purely from a forward-facing depth camera — no geometric shortcuts, no hand-coded repulsion from ground-truth positions.

---

## Repository structure

```
├── drone_obstacle_env.py   # CosysAirSim Gymnasium environment
├── sac_multitask.py        # SAC agent (CNN encoder + actor + twin-Q critics)
├── train.py                # Training and evaluation entry point
├── real_world_agent.py     # Real-hardware inference loop (hardware-agnostic template)
├── checkpoints/
│   └── latest.pth          # Trained weights (curriculum level 6: 5 obstacles, 0.5 m threshold)
└── logs/                   # CSV training logs (auto-created)
```

---

## Method overview

### Observation space

The agent receives three inputs at every 50 Hz control step:

| Key | Shape | Description |
|-----|-------|-------------|
| `depth` | `(4, 64, 64)` float32 | 4 stacked forward depth frames, normalised to [0,1] (near=1, far=0) |
| `vec_hist` | `(36,)` float32 | 3 stacked proprioceptive frames, each 12-dim: `[goal_body \| vel_body \| omega_body \| acc_body]` |
| `prev_action` | `(4,)` float32 | Last action in [-1, 1] |

All body-frame vectors are computed by rotating world-frame kinematics with the orientation quaternion. The heading reward keeps the depth camera pointed toward the goal, ensuring obstacle information is always in the FOV.

### Action space

Four continuous outputs in [-1, 1], scaled before sending to the flight controller:

| Output | Physical command | Range |
|--------|-----------------|-------|
| `action[0]` | Roll | ±60° |
| `action[1]` | Pitch | ±60° |
| `action[2]` | Yaw rate | ±180°/s |
| `action[3]` | Throttle | 0 → max |

### Agent architecture

```
Depth (4×64×64)
      │
   CNNEncoder                   Three conv layers (32→64→64)
      │                         then Linear → LayerNorm → Tanh
      ▼
  feat (256)
      │
  cat with vec_hist (36) + prev_action (4)
      │
  state (296)
      ├──▶  Actor   MLP 256→256→256→128 → mean + log_std → tanh-squashed action
      └──▶  Critic  twin-Q MLP 256→256→256→128→1
```

Standard SAC with automatic entropy tuning. No LSTM, no pose estimator — the proprioceptive history stack provides sufficient temporal context while keeping the replay buffer simple and memory-efficient.

### Reward

```
r = w_approach · dot(vel, goal_dir) · tanh(dist / d_brake)   approach velocity with braking
  + w_accel   · dot(acc, goal_dir) · tanh(dist / d_brake)   radial acceleration
  + w_dist    · exp(-k_dist · dist²)                         distance bowl
  + w_heading · goal_body_unit[0]                            heading alignment
  - 5 · z_err²                                               altitude correction
  - w_depth   · (depth_excess ^ sharpness)                   CNN-based proximity penalty
  + time_penalty                                             per-step cost
```

Terminal events: success +500, collision −1000, out-of-bounds −500.

The depth proximity penalty is derived from the depth image, not ground-truth geometry. This forces the CNN encoder to learn obstacle structure rather than being bypassed by a separate geometric signal.

### Curriculum

| Level | Obstacles | Ring(s) | Reach threshold |
|-------|-----------|---------|-----------------|
| 0 | 0 | — | 5.0 m |
| 1 | 1 | 1 (r = 7 m) | 4.0 m |
| 2 | 2 | 1 | 3.0 m |
| 3 | 3 | 1 | 2.0 m |
| 4 | 4 | 1 | 1.5 m |
| 5 | 5 | 1 | 1.0 m |
| **6** | **5** | **1** | **0.5 m** ← trained state |
| 7 | 10 | 2 (r = 7, 14 m) | 0.5 m |
| 8 | 15 | 3 (r = 7, 14, 21 m) | 0.5 m |

Obstacles are placed on concentric rings around the goal, each ring with a randomised 75° angular gap. The drone must penetrate the ring cluster — detouring around the outside is not possible because the goal is at the centre. Levels 7–8 are fine-tuning stages added after the baseline policy converged at level 6.

---

## Requirements

```
Python >= 3.9
torch >= 2.0
gymnasium >= 0.29
numpy
opencv-python
cosysairsim          # for simulation training/testing only
```

Install:

```bash
pip install torch gymnasium numpy opencv-python
# CosysAirSim Python client:
pip install cosysairsim
```

---

## Simulation training

CosysAirSim must be running and a multirotor vehicle must be present before starting.

```bash
# Start training from scratch
python train.py train

# Resume from the last checkpoint (default behaviour)
python train.py train --resume

# Key options
python train.py train \
    --max-episodes 50000 \
    --max-steps    1000  \
    --batch-size   256   \
    --buffer-size  100000\
    --checkpoint-dir checkpoints \
    --log-dir        logs
```

The curriculum advances automatically when the rolling 50-episode success rate exceeds 40%. To start fine-tuning from the supplied checkpoint (level 6), run with `--resume` — the agent will load `checkpoints/latest.pth` and continue from where training left off.

---

## Simulation evaluation

```bash
# Evaluate the trained policy at the trained level (level 6, default)
python train.py eval --checkpoint checkpoints/latest.pth

# Evaluate at a specific level
python train.py eval --checkpoint checkpoints/latest.pth \
    --max-steps 1000 --eval-episodes 20
```

---

## Real-world deployment

`real_world_agent.py` is a **hardware-agnostic inference template**. It contains the complete observation pipeline and control loop, but the hardware layer — camera read, state read, and command send — is left as stubs that you replace with calls specific to your platform and experiment setup.

### What you implement

Open `real_test_ex.py` and find `class DroneHardware` (Section 1, near the top). Implement the five methods for your hardware:

```python
class DroneHardware:

    def get_depth_image(self) -> np.ndarray:
        # Return (H, W) float32 array of planar depth in metres.
        # Any resolution — the pipeline resizes to 64×64 internally.
        raise NotImplementedError

    def get_state(self) -> dict:
        # Return dict with keys:
        #   pos_world  (3,) — position in NED metres
        #   vel_world  (3,) — velocity in m/s, NED world frame
        #   acc_world  (3,) — linear acceleration, NED world frame, GRAVITY REMOVED
        #   ori_quat   (4,) — orientation [w, x, y, z]
        #   omega_body (3,) — angular velocity rad/s, BODY FRAME
        raise NotImplementedError

    def send_command(self, roll, pitch, yaw_rate, throttle):
        # Send scaled physical commands to your flight controller.
        # roll / pitch : radians, ±60°
        # yaw_rate     : rad/s,   ±180°/s
        # throttle     : float,   [0.0, 1.0] — map to your FC's scale
        raise NotImplementedError

    def send_stop(self):
        # Command the drone to hold position immediately.
        # Called on Ctrl-C, watchdog trigger, or any unhandled exception.
        raise NotImplementedError

    def close(self):
        # Release resources, close connections.
        pass
```

Everything outside `DroneHardware` — the observation pipeline, normalisation, action scaling, 50 Hz loop, safety watchdog, and signal handling — is fixed and shouldn't be changed, as it must match the training environment exactly.

### Running

```bash
python real_world_agent.py \
    --checkpoint checkpoints/latest.pth \
    --goal 10.0 0.0 -1.5       # NED metres from your fixed origin
```

With safety bounds (NED metres, defaults match the training arena):

```bash
python real_world_agent.py \
    --checkpoint checkpoints/latest.pth \
    --goal 10.0 0.0 -1.5       \
    --x-min -15  --x-max 15    \
    --y-min -15  --y-max 15    \
    --z-min -10  --z-max -0.5
```

Deterministic mode (default) is recommended for final experiments.

### Important notes for real-world use

**Coordinate frame.** All vectors must be in NED (North=+X, East=+Y, Down=+Z), the same convention used throughout training. Goal Z is negative for altitude above ground.

**Gravity removal.** `acc_world` must be the net aerodynamic acceleration with gravity already subtracted. In AirSim this comes from `kinematics_estimated.linear_acceleration`, which is gravity-free. On real hardware you typically compute it as `R @ imu_acc_body - [0,0,g]` where `g = 9.81 m/s²`. Using raw IMU acceleration (which includes gravity) will silently corrupt the `acc_body` component of the proprioceptive state.

**Control frequency.** The loop targets 50 Hz. If your sensor stack cannot sustain 50 Hz the policy will still run, but performance will degrade because the training assumed 20 ms steps. The runner logs a warning whenever a cycle overruns by more than 10 ms.

**Buffer priming.** The runner collects 4 real depth frames before releasing control, so the CNN starts with actual scene information rather than zeros.

**Safety watchdog.** Set `--x-min/max`, `--y-min/max`, `--z-min/max` to your physical safe volume. The drone stops immediately if it exits those bounds or exceeds 15 m/s ground speed.

---

## Checkpoint

`checkpoints/latest.pth` contains the policy trained. The file stores encoder, actor, critic, and both target networks, plus `train_steps` and the `SACConfig` used during training.

To load weights manually:

```python
from sac_multitask import SAC, SACConfig

agent = SAC(frame_stack=4, image_size=64, action_dim=4, vec_hist_dim=36)
agent.load("checkpoints/latest.pth")

action = agent.select_action(obs, deterministic=True)
```
