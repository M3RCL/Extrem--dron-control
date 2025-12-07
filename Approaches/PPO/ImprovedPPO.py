from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math
import time
import random
import numpy as np
from collections import deque

import math
import time
import random
import numpy as np
from collections import deque
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import airsim
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
class ImprovedDroneFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor with better architecture and normalization."""
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(ImprovedDroneFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Add input normalization
        self.depth_normalizer = nn.BatchNorm2d(1)
        self.feature_normalizer = nn.BatchNorm1d(observation_space.spaces["features"].shape[0])
        
        # Improved CNN for depth image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten()
        )
        
        # Improved MLP for vector features with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(observation_space.spaces["features"].shape[0], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)  # Add dropout for regularization
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observations):
        # Normalize depth images
        depth = observations["depth"]
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)
        depth = self.depth_normalizer(depth)
        
        # Process depth through CNN
        cnn_features = self.cnn(depth)
        
        # Normalize and process vector features
        vector_features = self.feature_normalizer(observations["features"])
        mlp_features = self.mlp(vector_features)
        
        # Combine features
        combined = torch.cat((cnn_features, mlp_features), dim=1)
        return self.fusion(combined)
class DroneEnv(gym.Env):
    """Enhanced Drone Environment with updated actions and heading reward."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, waypoint_list=None, max_episode_steps=1000, client_id=0, render_mode='human'):
        super(DroneEnv, self).__init__()
        
        # Use different client IDs for parallel environments
        self.client_id = client_id
        self.render_mode = render_mode
        self.thresh_dist = 45.0
        self.beta = 1.0
        self.speed_weight = 0.5
        self.min_speed = -0.05
        self.max_speed = 50.0
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception as e:
            print(f"Failed to connect to AirSim (client {client_id}): {str(e)}")
            raise
        
        self.step_length = 25.0
        # Updated action space: 
        # 0: move forward (+x)
        # 1: move right (+y)
        # 2: move up (+z)
        # 3: move left (-y)
        # 4: move down (-z)
        # 5: rotate left (yaw +)
        # 6: rotate right (yaw -)
        self.action_space = spaces.Discrete(7)
        
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=np.array([-25, -math.pi, -25, -25, -25, -10, -10, -10], dtype=np.float32),
                high=np.array([ 25,  math.pi,  25,  25,  25,  10,  10,  10], dtype=np.float32)
            ),
            "depth": spaces.Box(
                low=0.0, 
                high=100.0, 
                shape=(84,84), 
                dtype=np.float32
            )
        })

        self.speed = 10.0
        self.dt = 0.1
        #np.array([  1.55265,   -38.9786,    -25.5225   ]),
        if waypoint_list is None:
            self.waypoints = [
                np.array([  12.87350317, -52.07368802, -36.06586883]),
                np.array([  38.98696329, -60.61485177, -57.63369411]),
                np.array([  68.69143613, -67.10846262, -62.36151646]),
                np.array([ 100.89062751, -69.06089195, -60.88487649]),
                np.array([ 134.48824319, -66.97851115, -55.83931483]),
                np.array([ 168.38798898, -61.3676916,  -49.86037208]),
                np.array([ 201.49670725, -52.73448285, -45.57539505]),
                np.array([ 233.1305958,  -41.54293266, -44.54235545]),
                np.array([ 263.41745251, -28.1738148,  -46.19905066]),
                np.array([ 292.58067796, -12.99809375, -49.73353133]),
                np.array([ 320.84367277,   3.613266,    -54.33384812]),
                np.array([ 348.42983753,  21.28929999, -59.18805166]),
                np.array([ 375.56257285,  39.65904373, -63.4841926 ]),
                np.array([ 402.46527931,  58.35153273, -66.4103216 ]),
                np.array([ 429.36135754,  76.99580253, -67.15448929]),
                np.array([ 456.47420812,  95.22088863, -64.90474633]),
                np.array([ 484.02723165, 112.65582657, -58.84914337]),
                np.array([ 512.24382875, 128.92965185, -48.17573104]),
                np.array([ 541.3474,     143.6714,     -32.07256   ])
            ]
        else:
            self.waypoints = waypoint_list

        self.current_wp_index = 0
        self.goal_threshold = 5.0 # waypoint bonus threshold
        self.obstacle_threshold = 3.0
        self.prev_distance = None
        self.step_count = 0
        self.max_episode_steps = max_episode_steps
        self.waypoint_start_step = 0
        self.reward = 0
        self.waypoint_start_dist = np.array([0.0, 0.0, 0.0])
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.successful_waypoints = 0
        
        self.np_random = None
        self.seed()
        
        # Initialize previous velocity for smoothness penalty
        self.prev_velocity = np.array([0.0, 0.0, 0.0])
        
    def _randomize_environment(self):
        self.client.simSetPhysicsEngineParameter(mass=random.uniform(0.8, 1.2))
        wind = airsim.Vector3r(
            random.uniform(-2, 2),
            random.uniform(-2, 2),
            random.uniform(-0.5, 0.5)
        )
        self.client.simSetWind(wind)
        
    def _shape_reward(self, base_reward, state, info):
        shaped_reward = base_reward
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        smooth_flight_reward = -0.1 * (abs(velocity.x_val) + abs(velocity.y_val) + abs(velocity.z_val))
        shaped_reward += smooth_flight_reward
        energy_penalty = -0.05 * (self.speed ** 2)
        shaped_reward += energy_penalty
        shaped_reward *= (1.0 + 0.2 * getattr(self, 'curriculum_level', 0))
        return shaped_reward

    def seed(self, seed=None):
        super().reset(seed=seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_wp_index = 0
        self.waypoint_start_step = 0
        self.early_terminate()
        try:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            start_pose = airsim.Pose()
            start_pose.position.x_val = 12.87350317-5
            start_pose.position.y_val = -52.07368802+ 3
            start_pose.position.z_val = -36.06586883
            
            self.client.simSetVehiclePose(start_pose, True)
            
            time.sleep(0.1)
            self.client.takeoffAsync().join()
            state = self._get_state()
            self.prev_distance = self._distance_to_goal(state["features"])
            self.waypoint_start_dist = self.prev_distance
            self.waypoint_start_step = self.step_count
            # Reset previous velocity to zero at the beginning of each episode.
            self.prev_velocity = np.array([0.0, 0.0, 0.0])
            return state, {}
        except Exception as e:
            print(f"Error in reset (client {self.client_id}): {str(e)}")
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                return self.reset(seed=seed, options=options)
            except:
                raise RuntimeError(f"Failed to reset environment (client {self.client_id})")

    def step(self, action):
        self.step_count += 1
        
        # Actions 0-4: translation; 5: rotate left; 6: rotate right.
        if action < 5:
            quad_offset = self.interpret_action(action)
            quad_state = self.client.getMultirotorState()
            quad_vel = quad_state.kinematics_estimated.linear_velocity
            vx = quad_vel.x_val if not math.isnan(quad_vel.x_val) else 0.0
            vy = quad_vel.y_val if not math.isnan(quad_vel.y_val) else 0.0
            vz = quad_vel.z_val if not math.isnan(quad_vel.z_val) else 0.0
            if math.isnan(quad_vel.x_val) or math.isnan(quad_vel.y_val) or math.isnan(quad_vel.z_val):
                print(f"[DEBUG] NaN detected in linear velocity in step(): Original values: {quad_vel}")
            try:
                self.client.moveByVelocityAsync(
                    vx + self.step_length * quad_offset[0],
                    vy + self.step_length * quad_offset[1],
                    vz + self.step_length * quad_offset[2],
                    0.7
                ).join()
            except Exception as e:
                print(f"Error applying translation action: {e}")
                return self._get_state(), -100, True, True, {}
        else:
            yaw_rate = 30 if action == 5 else -30
            try:
                self.client.rotateByYawRateAsync(yaw_rate, 1.5).join()
            except Exception as e:
                print(f"Error applying rotation action: {e}")
                return self._get_state(), -100, True, True, {}

        state = self._get_state()
        
        # Use our new reward function
        reward = self._compute_enhanced_reward(state)
        
        terminated = False
        truncated = False
        if self.client.simGetCollisionInfo().has_collided:
            reward =  -100.0
            self.collision_count += 1
            print("Collision detected!")
            terminated = True
        # If reward is very low, terminate the episode
        if self.early_terminate():
            terminated = True
            print("Way away from point reward", reward)
        

        curr_distance = self._distance_to_goal(state["features"])
        
        if self.current_wp_index >= len(self.waypoints):
                print("All waypoints reached!")
                terminated = True
        
        if self.step_count >= self.max_episode_steps:
            truncated = True
        #print("reward", reward)

        info = {
            "min_depth": float(np.min(state["depth"])),
            "collisions": self.collision_count,
            "waypoints_reached": self.successful_waypoints,
            "distance": curr_distance
        }
        return state, reward, terminated, truncated, info
    def early_terminate(self):
        pose = self.client.simGetVehiclePose()
        
        current_pos = np.array([
            pose.position.x_val,
            pose.position.y_val,
            pose.position.z_val
        ])
        dist =1e7
        #starting_pos = self.waypoint_start_dist
        next_wp = self.waypoints[self.current_wp_index]
        dist = min(dist,np.linalg.norm(np.cross((current_pos - next_wp), (current_pos - self.waypoint_start_dist))/np.linalg.norm(next_wp - self.waypoint_start_dist)))
        #print("dist", dist)
        if dist > self.thresh_dist:
            return True
        return False

    def _compute_enhanced_reward(self, state):
        """
        New reward function with multiple components:
          1. Distance penalty (closer to the waypoint is better)
          2. Bonus reward if within a threshold (waypoint reached)
          3. Smoothness penalty (penalize abrupt accelerations)
          4. Collision penalty
          5. Time penalty
          6. Small heading alignment reward
        """
        # Get current drone position
        pose = self.client.simGetVehiclePose()
        
        current_pos = np.array([
            pose.position.x_val,
            pose.position.y_val,
            pose.position.z_val
        ])
        next_wp = self.waypoints[self.current_wp_index]
        
        distance = np.linalg.norm(current_pos - next_wp)
        #print("distance", distance)
        
        # If the drone is far away, return an immediate penalty
        
        
        # Reward weights (tune these as needed)
        alpha = 1   # Distance weight
        beta = 2.0    # Waypoint bonus
        gamma = 0.01   # Smoothness penalty weight
        delta = 0.3 # dis upwardas goal
        #distz = state["features"][2]
        epsilon = 2  # Time penalty
        dist = 1e7
        goal_direction = next_wp - current_pos
        if distance > 1e-6:  # Avoid division by zero
            goal_direction_normalized = goal_direction / distance
        else:
            goal_direction_normalized = np.zeros_like(goal_direction)
        if self.current_wp_index < len(self.waypoints)-1:
            af_nex_wp = self.waypoints[self.current_wp_index + 1]
            dist = min(dist, np.linalg.norm(np.cross((af_nex_wp - next_wp), (af_nex_wp - self.waypoint_start_dist))/np.linalg.norm(next_wp - af_nex_wp)))
        
        else:
        
            dist = min(dist, np.linalg.norm(np.cross((current_pos - next_wp), (current_pos - self.waypoint_start_dist))/np.linalg.norm(next_wp - self.waypoint_start_dist)))
        reward = beta* math.exp(-alpha * dist) 
        if distance > self.thresh_dist:
            reward += -50
        if distance < self.goal_threshold:
            reward += 150.0
            self.successful_waypoints += 1
            self.waypoint_start_dist = distance
            print("Waypoint reached!")
            self.current_wp_index += 1
        # 1. Distance reward (the closer, the better)
        #reward = alpha / (distance +1)+delta
        #reward +=0.01 * ((distz))
        # 2. Waypoint bonus if within a threshold (using goal_threshold)
        
        
        #reward += delta * np.sum((current_pos - next_wp) / self.dt)
        # 3. Smoothness penalty: penalize high acceleration
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        current_velocity = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        goal_direction = next_wp - current_pos
        
            
        velocity_toward_goal = np.dot(current_velocity, goal_direction_normalized)
        reward += beta * velocity_toward_goal
        acceleration = (current_velocity - self.prev_velocity) / self.dt
        smooth_penalty = gamma * np.linalg.norm(acceleration)
        reward -= smooth_penalty
        # Update previous velocity for next step
        self.prev_velocity = current_velocity.copy()
        
        # 4. Collision penalty
       
        
        # 5. Time penalty
        reward -= epsilon * self.dt
        
        # 6. Small reward for heading alignment
        heading_error = state["features"][1]
        heading_reward = 0.1 * math.cos(heading_error)
        reward += heading_reward
        
        return reward 

    def interpret_action(self, action):
        # Actions:
        # 0: forward (+x)
        # 1: right (+y)
        # 2: up (+z)
        # 3: left (-y)
        # 4: down (-z)
        if action == 0:
            return (1, 0, 0)
        elif action == 1:
            return (0, 1, 0)
        elif action == 2:
            return (0, 0, 1)
        elif action == 3:
            return (0, -1, 0)
        elif action == 4:
            return (0, 0, -1)
        else:
            return (0, 0, 0)  # For translation; rotation handled separately

    def _get_state(self):
        pose = self.client.getMultirotorState().kinematics_estimated
        drone_x = pose.position.x_val
        drone_y = pose.position.y_val
        drone_z = pose.position.z_val

        quad_state = self.client.getMultirotorState()
        quad_vel = quad_state.kinematics_estimated.linear_velocity
        quad_ang = quad_state.kinematics_estimated.angular_velocity

        vx = quad_vel.x_val if not math.isnan(quad_vel.x_val) else 0.0
        vy = quad_vel.y_val if not math.isnan(quad_vel.y_val) else 0.0
        vz = quad_vel.z_val if not math.isnan(quad_vel.z_val) else 0.0
        if math.isnan(quad_vel.x_val) or math.isnan(quad_vel.y_val) or math.isnan(quad_vel.z_val):
            print(f"[DEBUG] NaN detected in _get_state() linear velocity: {quad_vel}")
        
        drone_yaw = self._get_yaw_from_pose(pose)
        goal = self.waypoints[self.current_wp_index]
        goal_x, goal_y, goal_z = goal
        pos = np.array([drone_x, drone_y, drone_z])
        goal = np.array([goal_x, goal_y, goal_z])
        dx = goal_x - drone_x
        dy = goal_y - drone_y
        distance = np.linalg.norm(pos- goal)
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - drone_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        altitude_error = goal_z - drone_z

        cos_yaw = math.cos(-drone_yaw)
        sin_yaw = math.sin(-drone_yaw)
        vx_body = vx * cos_yaw - vy * sin_yaw
        vy_body = vx * sin_yaw + vy * cos_yaw
        vz_body = vz

        roll_rate = quad_ang.x_val
        pitch_rate = quad_ang.y_val
        yaw_rate = quad_ang.z_val

        hybrid_features = np.array([
            distance,
            heading_error,
            #altitude_error,
            vx_body, vy_body, vz_body,
            roll_rate, pitch_rate, yaw_rate
        ], dtype=np.float32)

        depth_img = self._get_depth_image()
        return {"features": hybrid_features, "depth": depth_img}

    def _distance_to_goal(self, features):
        return features[0]

    def _get_yaw_from_pose(self, pose):
        q = pose.orientation
        siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _get_depth_image(self):
        try:
            request = airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)
            responses = self.client.simGetImages([request])
            if (not responses or responses[0] is None or responses[0].width == 0 or responses[0].height == 0):
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
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass

def make_env(rank, seed=0):
    def _init():
        env = DroneEnv()
        env.seed(seed + rank)
        return Monitor(env)
    return _init
class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )
    
    def forward(self, x):
        return F.relu(x + self.block(x))

class ImprovedDroneEnv(DroneEnv):
    """Enhanced drone environment with better rewards and features."""
    def __init__(self, *args, **kwargs):
        super(ImprovedDroneEnv, self).__init__(*args, **kwargs)
        
        # Add observation normalization
        self.running_mean = np.zeros(9)
        self.running_std = np.ones(9)
        self.count = 0
        
        # Additional metrics
        self.smoothness_buffer = deque(maxlen=10)
        self.progress_buffer = deque(maxlen=100)
    
    def _compute_enhanced_reward(self, state):
        base_reward = super()._compute_enhanced_reward(state)
        
        # Add smoothness reward
        quad_state = self.client.getMultirotorState()
        current_vel = np.array([
            quad_state.kinematics_estimated.linear_velocity.x_val,
            quad_state.kinematics_estimated.linear_velocity.y_val,
            quad_state.kinematics_estimated.linear_velocity.z_val
        ])
        
        if len(self.smoothness_buffer) > 0:
            vel_diff = np.linalg.norm(current_vel - self.smoothness_buffer[-1])
            smoothness_reward = -0.1 * vel_diff
        else:
            smoothness_reward = 0
        
        self.smoothness_buffer.append(current_vel)
        
        # Add progress reward
        current_pos = np.array([
            quad_state.kinematics_estimated.position.x_val,
            quad_state.kinematics_estimated.position.y_val,
            quad_state.kinematics_estimated.position.z_val
        ])
        
        if len(self.progress_buffer) > 0:
            progress = np.linalg.norm(current_pos - self.progress_buffer[-1])
            progress_reward = 0.1 * progress
        else:
            progress_reward = 0
            
        self.progress_buffer.append(current_pos)
        
        # Combine rewards
        total_reward = base_reward + smoothness_reward + progress_reward
        
        # Add exploration bonus
        if self._is_exploring_new_area(current_pos):
            total_reward += 1.0
            
        return total_reward
    
    def _is_exploring_new_area(self, current_pos):
        # Check if current position is far from previously visited positions
        return not any(np.linalg.norm(current_pos - prev_pos) < 2.0 
                      for prev_pos in self.progress_buffer)

def train_improved_ppo(total_timesteps=1e6):
    """Enhanced PPO training with better hyperparameters and features."""
    env = ImprovedDroneEnv()
    env = Monitor(env)
    
    policy_kwargs = dict(
        features_extractor_class=ImprovedDroneFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 256, 128],  # Deeper policy network
            vf=[256, 256, 128]   # Deeper value network
        ),
        activation_fn=nn.ReLU,
        ortho_init=True
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_drone_logs/",
        learning_rate=linear_schedule(3e-4),  # Use learning rate scheduling
        batch_size=512,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.2),
        clip_range_vf=linear_schedule(0.2),
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        n_epochs=10
    )
    
    # Add callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./checkpoints/",
        name_prefix="drone_model"
    )
    
    eval_callback = EvalCallback(
        eval_env=ImprovedDroneEnv(),
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    return model, [checkpoint_callback, eval_callback]

def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


if __name__ == "__main__":
    model, callbacks = train_improved_ppo()
    model.learn(
        total_timesteps=1e6,
        callback=callbacks,
        progress_bar=True
)
