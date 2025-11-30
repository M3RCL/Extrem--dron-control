# X500 Drone RL Simulation

Custom Ignition Gazebo + ROS2 Humble simulation for training RL agents on quadcopter control, using the PX4 X500 drone model without PX4/QGroundControl dependencies.

## Features

- ✅ Direct motor control (no flight controller needed)
- ✅ Gymnasium-compatible environment for RL training
- ✅ Real-time state observation (IMU, GPS, Barometer)
- ✅ Customizable reward functions
- ✅ SAC (Soft Actor-Critic) example implementation
- ✅ Modular architecture for easy experimentation

## Architecture

```
┌─────────────────┐
│  RL Agent       │
│  (SAC/PPO/TD3)  │
└────────┬────────┘
         │ actions [thrust, roll, pitch, yaw]
         ▼
┌─────────────────┐
│ Gym Environment │ ◄──── state [pos, vel, orientation]
│ (drone_gym_env) │
└────────┬────────┘
         │ ROS2 topics
         ▼
┌─────────────────┐
│ Drone Controller│ ◄──── /x500/state
│ (motor mixing)  │ ────► /x500/action
└────────┬────────┘
         │ motor speeds
         ▼
┌─────────────────┐
│ Ignition Gazebo │
│ (X500 model)    │
└─────────────────┘
```

## Prerequisites

```bash
# Ubuntu 22.04 with ROS2 Humble
sudo apt update

# Install ROS2 Humble (if not already installed)
# Follow: https://docs.ros.org/en/humble/Installation.html

# Install Gazebo (Ignition)
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-sim ros-humble-ros-gz-bridge

# Install dependencies
sudo apt install python3-pip python3-colcon-common-extensions
pip3 install gymnasium stable-baselines3 numpy scipy torch
```


### Setup model path

```bash
# Add to ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:~/drone_rl_ws/src/x500_simulation/models' >> ~/.bashrc
source ~/.bashrc
```

### Build

```bash
cd ~/drone_rl_ws
colcon build
source install/setup.bash
```

## Usage

### Launch Simulation

```bash
# Terminal 1: Launch Gazebo + Controller
ros2 launch x500_simulation drone_sim.launch.py

# The simulation will start with:
# - Gazebo visualization
# - X500 drone spawned at origin
# - Drone controller node running
# - ROS2-Gazebo bridge active
```

### Train RL Agent

```bash
# Terminal 2: Start training
cd ~/drone_rl_ws/src/x500_simulation/scripts
python3 train_drone.py --mode train --timesteps 1000000


### Test Trained Agent

```bash
python3 train_drone.py --mode test --model_path ./models/drone_sac_final.zip
```

### Manual Testing

```bash
# Publish action directly
ros2 topic pub /x500/action std_msgs/msg/Float32MultiArray \
  "data: [0.5, 0.0, 0.0, 0.0]" --once

# Monitor state
ros2 topic echo /x500/state

# Monitor odometry
ros2 topic echo /model/x500/odometry
```

## Customization

### Modify Reward Function

Edit `drone_gym_env.py`:

```python
def compute_reward(self):
    # Your custom reward logic
    distance = np.linalg.norm(self.state[:3] - self.target_position)
    
    # Example: sparse reward
    reward = 100.0 if distance < 0.5 else -distance
    
    return reward
```

### Change Motor Mixing

Edit `drone_controller.py`:

```python
def compute_motor_speeds(self, thrust, roll_rate, pitch_rate, yaw_rate):
    # Custom mixing matrix for different drone configurations
    mixing_matrix = np.array([
        # [thrust, roll, pitch, yaw]
        [1,  -1,  1,  1],  # Modify based on your setup
        # ...
    ])
```


## State Space

The environment provides 15D observations:

```python
[
    # Position (3D)
    x, y, z,
    
    # Velocity (3D)
    vx, vy, vz,
    
    # Orientation (3D, Euler angles)
    roll, pitch, yaw,
    
    # Angular velocity (3D)
    wx, wy, wz,
    
    # Target position (3D)
    target_x, target_y, target_z
]
```

## Action Space

4D continuous actions:

```python
[
    thrust,      # [0, 1] - normalized thrust
    roll_rate,   # [-1, 1] - desired roll rate (rad/s)
    pitch_rate,  # [-1, 1] - desired pitch rate (rad/s)
    yaw_rate     # [-1, 1] - desired yaw rate (rad/s)
]
```



- [Gazebo Documentation](https://gazebosim.org/docs)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
