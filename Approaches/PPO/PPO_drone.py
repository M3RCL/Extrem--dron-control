#!/usr/bin/env python3
"""
PPO Training Script for Aggressive Drone Navigation with Full Depth Image Input

This script defines a custom Gym environment for a multirotor drone in Colosseum.
The observation returned is a dictionary with two keys:
  - "features": a 9-dimensional vector 
      [dx_body, dy_body, dz_error, vx_body, vy_body, vz_body, roll_rate, pitch_rate, yaw_rate]
  - "depth": an 84×84 depth image from a DepthPerspective camera
The action space consists of 8 discrete actions:
  0: move +x (forward)
  1: move +y (right)
  2: move +z (up)
  3: move -x (backward)
  4: move -y (left)
  5: move -z (down)
  6: increase speed
  7: decrease speed
The reward function is designed to encourage rapid progress toward waypoints,
good heading alignment, and quick braking near obstacles.
A custom feature extractor is implemented to process the depth image (via CNN)
and the low-dimensional features (via an MLP). PPO from stable-baselines3 is then
used to train the agent.
"""
import math
import time
import random
import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import airsim
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# =========================
# Custom Gym Environment
# =========================

class DroneEnv(gym.Env):
    """Enhanced version of the DroneEnv with Gymnasium compatibility"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, waypoint_list=None, max_episode_steps=100000, client_id=0, render_mode='human'):
        super(DroneEnv, self).__init__()
        
        # Use different client IDs for parallel environments
        
        self.client_id = client_id
        self.render_mode = render_mode
        self.thresh_dist = 35.0  # Maximum distance threshold for reward calculation
        self.beta = 1.0  # Distance scaling factor
        self.speed_weight = 0.5  # Weight for speed reward
        self.min_speed = -500  # Minimum desired speed
        self.max_speed = 500.0  # Maximum desired speed
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception as e:
            print(f"Failed to connect to AirSim (client {client_id}): {str(e)}")
            raise
        
        self.step_length = 100.0
        self.action_space = spaces.Discrete(8)
        
        # Updated observation space: now 9-dimensional features instead of 4.
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=np.array([-50, -50, -50,   -25, -25, -25,   -10, -10, -10], dtype=np.float32),
                high=np.array([ 50,  50,  50,    25,  25,  25,    10,  10,  10], dtype=np.float32)
            ),
            "depth": spaces.Box(
                low=0.0, 
                high=100.0, 
                shape=(84,84), 
                dtype=np.float32
            )
        })

        # Speed parameters
        self.min_speed = -50.0
        self.max_speed = 50.0
        self.speed = 2.0
        self.dt = 0.1

        # Waypoint setup
        if waypoint_list is None:
            self.waypoints = [
            np.array([-8.55265, -31.9786, -19.0225]),
            np.array([48.59735, -63.3286, -60.07256]),
            np.array([193.5974, -55.0786, -46.32256]),
            np.array([369.2474, 35.32137, -62.5725]),
            np.array([541.3474, 143.6714, -32.07256]),
        ]
        else:
            self.waypoints = waypoint_list

        self.current_wp_index = 0
        self.goal_threshold = 2.0
        self.obstacle_threshold = 3.0
        self.prev_distance = None
        self.step_count = 0
        self.max_episode_steps = max_episode_steps
        self.waypoint_start_step = 0
        
        # Add episode statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.successful_waypoints = 0
        
        # Initialize random number generator
        self.np_random = None
        self.seed()     
        
    def _randomize_environment(self):
        """Add domain randomization for better robustness"""
        # Randomize drone mass
        self.client.simSetPhysicsEngineParameter(mass=random.uniform(0.8, 1.2))
        
        # Randomize wind conditions
        wind = airsim.Vector3r(
            random.uniform(-2, 2),
            random.uniform(-2, 2),
            random.uniform(-0.5, 0.5)
        )
        self.client.simSetWind(wind)
        
    def _shape_reward(self, base_reward, state, info):
        """Enhanced reward shaping for better learning"""
        shaped_reward = base_reward
        
        # Add smooth progression reward
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        smooth_flight_reward = -0.1 * (abs(velocity.x_val) + abs(velocity.y_val) + abs(velocity.z_val))
        shaped_reward += smooth_flight_reward
        
        # Add energy efficiency reward
        energy_penalty = -0.05 * (self.speed ** 2)
        shaped_reward += energy_penalty
        
        # Scale reward based on curriculum level (if used)
        shaped_reward *= (1.0 + 0.2 * getattr(self, 'curriculum_level', 0))
        
        return shaped_reward

    def seed(self, seed=None):
        """Set the seed for this env's random number generator(s)."""
        super().reset(seed=seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """Reset the environment with proper error handling."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.current_wp_index = 0
        self.waypoint_start_step = 0

        try:
            # Reset simulation and re-enable API control
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)

            # Set a random starting pose using seeded RNG
            start_pose = airsim.Pose()
            start_pose.position.x_val =  -10
            start_pose.position.y_val = -12
            start_pose.position.z_val = self.np_random.uniform(-20, -40)
            
            self.client.simSetVehiclePose(start_pose, True)
            time.sleep(0.1)  # Give AirSim time to process

            self.client.takeoffAsync().join()
            self.client.moveToZAsync(start_pose.position.z_val, 2).join()

            state = self._get_state()
            self.prev_distance = self._distance_to_goal(state["features"])
            self.waypoint_start_step = self.step_count
            
            return state, {}  # Return state and empty info dict for gymnasium compatibility

        except Exception as e:
            print(f"Error in reset (client {self.client_id}): {str(e)}")
            # Try to reconnect
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                return self.reset(seed=seed, options=options)
            except:
                raise RuntimeError(f"Failed to reset environment (client {self.client_id})")

    def step(self, action):
        """Execute one time step within the environment."""
        self.step_count += 1

        # Get the quadrotor offset based on the action
        quad_offset = self.interpret_action(action)
        quad_state = self.client.getMultirotorState()
        quad_vel = quad_state.kinematics_estimated.linear_velocity

        # Apply the actionb
        try:
            self.client.moveByVelocityAsync(
                quad_vel.x_val + self.speed * quad_offset[0],
                quad_vel.y_val + self.speed * quad_offset[1],
                quad_vel.z_val + self.speed * quad_offset[2],
                1
            ).join()
        except Exception as e:
            print(f"Error applying action: {e}")
            return self._get_state(), -100, True, True, {}

        # Get new state
        state = self._get_state()
        
        # Calculate reward using enhanced reward function
        reward = self._compute_enhanced_reward(state)
        
        # Check termination conditions
        terminated = False
        truncated = False
        if reward <= -15:
            terminated = True
        if self.client.simGetCollisionInfo().has_collided:
            reward = -100.0
            self.collision_count += 1
            print("Collision detected!")
            terminated = True
        reward = self._shape_reward(reward, state, {})

        # Check if we've reached the current waypoint
        curr_distance = self._distance_to_goal(state["features"])
        if curr_distance < self.goal_threshold:
            reward += 500.0/self.step_count  # Bonus for reaching waypoint
            self.successful_waypoints += 1
            print(f"Waypoint reached!")
            
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.waypoints):
                print("All waypoints reached!")
                terminated = True

        if self.step_count >= self.max_episode_steps:
            truncated = True
        

        info = {
            "min_depth": float(np.min(state["depth"])),
            "collisions": self.collision_count,
            "waypoints_reached": self.successful_waypoints,
            "distance": curr_distance
        }

        return state, reward, terminated, truncated, info
    def _compute_enhanced_reward(self, state):
        """
        Enhanced reward function that considers:
        1. Distance to the path between waypoints
        2. Current speed
        3. Progress toward goal
        """
        pts = [
            np.array([-8.55265, -31.9786, -19.0225]),
            np.array([48.59735, -63.3286, -60.07256]),
            np.array([193.5974, -55.0786, -46.32256]),
            np.array([369.2474, 35.32137, -62.5725]),
            np.array([541.3474, 143.6714, -32.07256]),
        ]
        # Get current position
        pose = self.client.simGetVehiclePose()
        current_pos = np.array([
            pose.position.x_val,
            pose.position.y_val,
            pose.position.z_val
        ])
        
        dist = 10000000
        for i in range(0, len(pts) - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((current_pos - pts[i]), (current_pos - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                ) 

        if dist > self.thresh_dist:
                reward = -15
        else:
                reward_dist = math.exp(-self.beta * dist) - 0.5
                quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
                reward_speed = ( np.linalg.norm([
            quad_vel.x_val,
            quad_vel.y_val,
            quad_vel.z_val
        ])
                    - 0.5
                )
                reward = reward_dist + reward_speed

        return reward

    def interpret_action(self, action):
        """
        Maps the discrete action to a directional offset and speed adjustment.
        Actions:
          0: move +x (forward)
          1: move +y (right)
          2: move +z (up)
          3: move -x (backward)
          4: move -y (left)
          5: move -z (down)
          6: increase speed
          7: decrease speed
        """
        if action == 0:
            offset = (1, 0, 0)
        elif action == 1:
            offset = (0, 1, 0)
        elif action == 2:
            offset = (0, 0, 1)
        elif action == 3:
            offset = (-1, 0, 0)
        elif action == 4:
            offset = (0, -1, 0)
        elif action == 5:
            offset = (0, 0, -1)
        elif action == 6:
            offset = (0, 0, 0)
            self.speed = min(self.speed + 1.0, self.max_speed)
        elif action == 7:
            offset = (0, 0, 0)
            self.speed = max(self.speed - 1.0, self.min_speed)
        else:
            offset = (0, 0, 0)
        return offset

    def _get_state(self):
        """
        Returns a dictionary observation with:
          - "features": a 9-dimensional vector 
              [dx_body, dy_body, dz_error, vx_body, vy_body, vz_body, roll_rate, pitch_rate, yaw_rate]
          - "depth": an 84x84 depth image (float32)
        """
        # Get current pose and waypoint differences.
        pose = self.client.simGetVehiclePose()
        drone_x = pose.position.x_val
        drone_y = pose.position.y_val
        drone_z = pose.position.z_val

        goal = self.waypoints[self.current_wp_index]
        goal_x, goal_y, goal_z = goal

        dx_world = goal_x - drone_x
        dy_world = goal_y - drone_y
        dz_world = goal_z - drone_z

        # Get drone yaw for transforming vectors into the body frame.
        drone_yaw = self._get_yaw_from_pose(pose)
        cos_yaw = math.cos(-drone_yaw)
        sin_yaw = math.sin(-drone_yaw)
        dx_body = dx_world * cos_yaw - dy_world * sin_yaw
        dy_body = dx_world * sin_yaw + dy_world * cos_yaw

        # Instead of using yaw error, we now include the drone's velocity and angular rates.
        quad_state = self.client.getMultirotorState()
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        quad_ang = quad_state.kinematics_estimated.angular_velocity

        # Transform global linear velocity to the drone's body frame.
        vx_global = quad_vel.x_val
        vy_global = quad_vel.y_val
        vz_global = quad_vel.z_val
        vx_body = vx_global * cos_yaw - vy_global * sin_yaw
        vy_body = vx_global * sin_yaw + vy_global * cos_yaw
        vz_body = vz_global

        # Angular rates (assumed to be in the body frame)
        roll_rate = quad_ang.x_val
        pitch_rate = quad_ang.y_val
        yaw_rate = quad_ang.z_val

        features = np.array([
            dx_body, dy_body, dz_world,
            vx_body, vy_body, vz_body,
            roll_rate, pitch_rate, yaw_rate
        ], dtype=np.float32)

        depth_img = self._get_depth_image()
        
        return {"features": features, "depth": depth_img}

    def _distance_to_goal(self, features):
        # Use only the position error components (indices 0 and 1).
        dx, dy = features[0], features[1]
        return math.sqrt(dx**2 + dy**2)

    def _get_yaw_from_pose(self, pose):
        q = pose.orientation
        siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _get_depth_image(self):
        """
        Retrieves the depth map from camera "3" (DepthPerspective),
        resizes it to 84x84, and returns a float32 numpy array.
        """
        try:
            request = airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)
            responses = self.client.simGetImages([request])
            if (not responses or responses[0] is None or 
                responses[0].width == 0 or responses[0].height == 0):
                return np.full((84,84), 100.0, dtype=np.float32)
            depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
            depth_img = depth_img.reshape(responses[0].height, responses[0].width)
            depth_img = cv2.resize(depth_img, (84,84), interpolation=cv2.INTER_LINEAR)
            return depth_img
        except Exception as e:
            print("Error reading depth image:", e)
            return np.full((84,84), 100.0, dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        """Cleanup environment properly."""
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = DroneEnv()
        env.seed(seed + rank)
        return Monitor(env)
    return _init

# =========================
# Custom Feature Extractor for PPO
# =========================

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DroneFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the DroneEnv.
    It processes a dictionary observation with:
      - "features": a 9-dimensional vector processed by an MLP.
      - "depth": an 84x84 depth image processed by a CNN.
    The outputs are concatenated and passed through a final layer.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(DroneFeatureExtractor, self).__init__(observation_space, features_dim)
        # Get shapes from observation space
        vector_shape = observation_space.spaces["features"].shape  # (9,)
        image_shape = observation_space.spaces["depth"].shape        # (84,84)

        # CNN for depth image: input will be (batch, 1, 84,84)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Determine output dimension of CNN by passing a dummy input
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(np.zeros((1, 1, 84, 84), dtype=np.float32))).shape[1]

        # MLP for vector features
        self.mlp = nn.Sequential(
            nn.Linear(vector_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Final layer: combine CNN and MLP outputs
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process depth image
        depth = observations["depth"]
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)  # add channel dimension to get (batch, 1, 84,84)
        cnn_out = self.cnn(depth)
        # Process vector features
        mlp_out = self.mlp(observations["features"])
        # Concatenate and process
        combined = torch.cat((cnn_out, mlp_out), dim=1)
        return self.linear(combined)


# =========================
# PPO Training with Custom Policy
# =========================

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_ppo(total_timesteps=200000):
    """Training function adjusted for single environment due to AirSim limitations"""
    
    # Create and wrap the environment
    env = DroneEnv()
    env = Monitor(env) 
    
    # Define network architecture
    policy_kwargs = dict(
        features_extractor_class=DroneFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(
            pi=[128, 256],
            vf=[128, 256]
        )
    )
    
    # Create PPO model with enhanced parameters
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_drone_logs/",
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        n_epochs=10
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./checkpoints/",
        name_prefix="drone_model"
    )
    
    # Train the model
    try:
        model.load(("./checkpoints/drone_model_220000_steps.zip"))
        print("Model loaded")
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        model.save("final_drone_model")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save("interrupted_drone_model")
    finally:
        env.close()
    
    return model

if __name__ == "__main__":
    ppo_model = train_ppo()
