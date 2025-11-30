#!/usr/bin/env python3
"""
Dual DQN Training Script for Aggressive Drone Navigation with Full Depth Image Input

This script modifies the original PPO implementation to use Dual DQN instead.
The observation and action spaces have been updated (with explicit yaw rotation and no backward action).
The training approach uses two Q-networks (current and target) to reduce overestimation bias.
"""
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

# Drone Environment with troubleshooting, heading reward, and updated action space (no backward action)
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
        
        self.step_length = 35.0
        # Updated action space: 
        # 0: move forward (+x)
        # 1: move right (+y)
        # 2: move up (+z)
        # 3: move left (-y)
        # 4: move down (-z)
        # 5: rotate left (yaw +)
        # 6: rotate right (yaw -)
        self.action_space = spaces.Discrete(6)
        
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=np.array([-15, -math.pi, -math.pi,   -math.pi], dtype=np.float32),
                high=np.array([ 15,  math.pi,  math.pi,    math.pi], dtype=np.float32)
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
    np.array([ 12.87350317, -52.07368802, -36.06586883]),
    np.array([ 36.13082460, -59.79192961, -56.52791401]),
    np.array([ 66.43108571, -66.76957152, -62.28243633]),
    np.array([ 98.06828455, -69.06830087, -61.18038764]),
    np.array([ 129.42899654, -67.52090755, -56.72173514]),
    np.array([ 160.34291208, -63.00117731, -51.22205306]),
    np.array([ 190.90583561, -55.83497013, -46.64255794]),
    np.array([ 221.03676980, -46.16065198, -44.57226201]),
    np.array([ 250.41959714, -34.22145192, -45.20389275]),
    np.array([ 278.84824209, -20.41131184, -47.87064589]),
    np.array([ 306.34814354,  -5.12075649, -51.87554520]),
    np.array([ 333.08684462,  11.30386012, -56.49261404]),
    np.array([ 359.30983407,  28.56461624, -61.01661208]),
    np.array([ 385.28315650,  46.38500521, -64.74038029]),
    np.array([ 411.24809179,  64.47019630, -66.93319638]),
    np.array([ 437.37017549,  82.46149602, -66.83556848]),
    np.array([ 463.67813611,  99.90397299, -63.71223223]),
    np.array([ 490.01109921,  116.25251218, -56.97515214]),
    np.array([ 516.03589875,  130.97690355, -46.38035276]),
    np.array([ 541.34740000,  143.67140000, -32.07256000]),
]

        else:
            self.waypoints = waypoint_list

        self.current_wp_index = 0
        self.goal_threshold = 4.0 # waypoint bonus threshold
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
        quad_offset = self.interpret_action(action)
        quad_state = self.client.getMultirotorState()
        quad_vel = quad_state.kinematics_estimated.linear_velocity
            
        # Smooth velocity transitions
        target_vx = self.step_length * quad_offset[0]
        target_vy = self.step_length * quad_offset[1]
        target_vz = self.step_length * quad_offset[2]
            
        current_vx = quad_vel.x_val if not math.isnan(quad_vel.x_val) else 0.0
        current_vy = quad_vel.y_val if not math.isnan(quad_vel.y_val) else 0.0
        current_vz = quad_vel.z_val if not math.isnan(quad_vel.z_val) else 0.0
            
        # Interpolate velocities
        alpha = 0.3  # Smoothing factor
        new_vx = current_vx + alpha * (target_vx - current_vx)
        new_vy = current_vy + alpha * (target_vy - current_vy)
        new_vz = current_vz + alpha * (target_vz - current_vz)
            
        try:
            self.client.moveByVelocityAsync(new_vx, new_vy, new_vz, 1).join()
        except Exception as e:
                print(f"Error applying translation action: {e}")
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
        pose = self.client.getMultirotorState().kinematics_estimated
        if self.current_wp_index < len(self.waypoints)-1:
            current_pos = np.array([
                pose.position.x_val,
                pose.position.y_val,
                pose.position.z_val
            ])
            if self.current_wp_index == len(self.waypoints):
                return False
            dist =1e7
            #starting_pos = self.waypoint_start_dist
            next_wp = self.waypoints[self.current_wp_index]
            af_nex_wp = self.waypoints[self.current_wp_index+1]
            dist = 10000000
            dist = min(
                        dist,
                        np.linalg.norm(np.cross((current_pos - next_wp), (current_pos - af_nex_wp)))
                        / np.linalg.norm(next_wp - af_nex_wp),
                    )
            if dist > self.thresh_dist:
                return True
        return False
    def compute_potential(self, current_pos, waypoint, path_start):
            # Project current_pos onto the line defined by path_start and waypoint
            line_vec = waypoint - path_start
            proj = path_start + np.dot(current_pos - path_start, line_vec) / np.dot(line_vec, line_vec) * line_vec
            return -np.linalg.norm(current_pos - proj)

        
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
        pose = self.client.getMultirotorState().kinematics_estimated
        drone_x = pose.position.x_val
        drone_y = pose.position.y_val
        drone_z = pose.position.z_val
        current_pos = np.array(list([drone_x, drone_y, drone_z]))
        next_wp = self.waypoints[self.current_wp_index]
        
        distance = np.linalg.norm(current_pos - next_wp)
        #print("distance", distance)
        
        # If the drone is far away, return an immediate penalty
        
        
        # Reward weights (tune these as needed)
        alpha = 1   # Distance weight
        beta = 1.0    # Waypoint bonus
        gamma = 0.01   # Smoothness penalty weight
        delta = 0.3 # dis upwardas goal
        #distz = state["features"][2]
        epsilon = 5  # Time penalty
        #dist = 1e7
        
        if self.current_wp_index < len(self.waypoints)-1:
            af_nex_wp = self.waypoints[self.current_wp_index + 1]
            #dist = min(dist, np.linalg.norm(np.cross((current_pos - next_wp), (-af_nex_wp + current_pos))/np.linalg.norm(next_wp - af_nex_wp)))
            potential_current = self.compute_potential(current_pos, next_wp, self.waypoint_start_dist)
            potential_next = self.compute_potential(af_nex_wp, next_wp, self.waypoint_start_dist)
            reward = 0
        #reward = beta * math.exp(-0.5 * dist) 
            dist = distance
            dist = min(
                    dist,
                    np.linalg.norm(np.cross((current_pos - next_wp), (current_pos - af_nex_wp)))
                    / np.linalg.norm(next_wp - af_nex_wp),
                )
            #dist = distance
            if dist < self.goal_threshold:
                reward += 150.0
                self.successful_waypoints += 1
                self.waypoint_start_dist = distance
                print("Waypoint reached!")
                self.current_wp_index += 1
            if dist > self.thresh_dist:
                reward = -15
            else:
                x , y, z =current_pos 
                gx , gy, gz = next_wp
                distx = np.linalg.norm(x - gx)
                disty = np.linalg.norm(y - gy)  
                distz = np.linalg.norm(z - gz)
                reward -=  math.exp(-beta * distx)
                reward -=  math.exp(-beta * disty)
                reward -=  math.exp(-beta * distz)
                quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
                current_velocity = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
                goal_direction = next_wp - current_pos
                goal_direction = next_wp - af_nex_wp
                if distance > 1e-6:  # Avoid division by zero
                    goal_direction_normalized = goal_direction / distance
                else:
                    goal_direction_normalized = np.zeros_like(goal_direction)
                    
                velocity_toward_goal = np.dot(current_velocity,  goal_direction_normalized)
                reward += beta * np.linalg.norm(velocity_toward_goal)
                acceleration = (current_velocity - self.prev_velocity) / self.dt
                smooth_penalty = gamma * np.linalg.norm(acceleration)
                reward -= beta* smooth_penalty
                # Update previous velocity for next step
                self.prev_velocity = current_velocity.copy()
                
                # 4. Collision penalty
            
                
                # 5. Time penalty
                reward -= epsilon * self.dt *10
                
                # 6. Small reward for heading alignment
                heading_error = (np.linalg.norm(
                        [
                            state["features"][1],
                            state["features"][2],
                            state["features"][3],
                        ]) - 5)
                #reward += 0.001 * heading_error
        else:
            potential_current = self.compute_potential(current_pos, next_wp, self.waypoint_start_dist)
            shaping_reward = 0.1* gamma * ( potential_current)
            reward = shaping_reward
        
        
        #reward += np.min(beta *(current_pos - next_wp) )
        
        # 1. Distance reward (the closer, the better)
        #reward = alpha / (distance +1)+delta
        #reward +=0.01 * ((distz))
        # 2. Waypoint bonus if within a threshold (using goal_threshold)
        
        
        #reward += delta * np.sum((current_pos - next_wp) / self.dt)
        # 3. Smoothness penalty: penalize high acceleration
        
        #reward += heading_reward
        
        return reward 

    def interpret_action(self, action):
        """Enhanced action interpretation with diagonal movements"""
        # Base movements
        movements = {
            0: (1, 0, 0),    # forward
            1: (-1, 0, 0), # left
            2: (0, 1, 0),    # right
            3: (0, 0, 1),    # up
            4: (0, 0, -1),   # down
            5: (0, -1, 0), # backward
            
            
            }
        return movements.get(action, (0, 0, 0))

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
        dz = goal_z - drone_z
        distance = np.linalg.norm(pos- goal)
        desired_heading = math.atan2(dy, dx)
        
        #heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        desired_pitch = math.atan2(dz, dx)
        desired_roll = math.atan2(dz, dy)
        heading_error = desired_heading - drone_yaw
        roll_error = desired_roll - self._get_roll_from_pose(pose)
        pitch_error = desired_pitch - self._get_pitch_from_pose(pose) # _get_roll,pitch_from_pose
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
            roll_error,
            pitch_error,
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
    def _get_roll_from_pose(self, pose):
        q = pose.orientation
        sinr_cosp = 2.0 * (q.w_val * q.x_val + q.y_val * q.z_val)
        cosr_cosp = 1.0 - 2.0 * (q.x_val**2 + q.y_val**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        return roll

    def _get_pitch_from_pose(self, pose):
        q = pose.orientation
        sinp = 2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
        # Clamp sinp to the range [-1, 1] to avoid errors from floating point imprecision.
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        return pitch

    def _get_depth_image(self):
        try:
            request = airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)
            responses = self.client.simGetImages([request])
            if (not responses or responses[0] is None or responses[0].width == 0 or responses[0].height == 0):
                return np.full((84,84), 100.0, dtype=np.float32)
            
            depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
            depth_img = depth_img.reshape(responses[0].height, responses[0].width)
            depth_img = cv2.resize(depth_img, (84,84), interpolation=cv2.INTER_LINEAR)
            #cv2.imshow("Depth", depth_img)
            return depth_img
        except Exception as e:
            print("Error reading depth image:", e)
            return np.full((84,84), 100.0, dtype=np.float32)

    def render(self, mode='human'):
        #
        cv2.waitKey(1)
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

# Larger DualDQNNetwork architecture (Dropout layers removed)
class DualDQNNetwork(nn.Module):

    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int):
        super(DualDQNNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LayerNorm([32, 20, 20]),  # Assuming 84x84 input
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            #nn.Dropout(0.2),  # Add dropout layer with 20% dropout rate
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.LayerNorm([128, 7, 7]),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(observation_space.spaces["features"].shape[0], 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            
            
            
            
            
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
        
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten+128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),  
            nn.LayerNorm(128),  
            nn.ReLU(),
            nn.Dropout(0.2)
            #nn.Dropout(0.2)  # Add dropout for regularization
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            #nn.Tanh()  # Bound advantage estimates
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.Tanh()  
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        training = self.training
        if not training:
            self.eval()
            
        depth = observation["depth"].unsqueeze(1)
        if depth.shape[-2:] != (84, 84):
            raise ValueError(f"Expected depth image of size 84x84, got {depth.shape[-2:]}")
            
        cnn_features = self.cnn(depth)
        vector_features = self.feature_net(observation["features"])
        
        # Normalize features before concatenation
        #cnn_features = F.normalize(cnn_features, p=2, dim=1)
        #vector_features = F.normalize(vector_features, p=2, dim=1)
        
        
        
        combined = torch.cat((cnn_features, vector_features), dim=1)
        fused_features = self.fusion(combined)
        
        advantage = self.advantage(fused_features)
        value = self.value(fused_features)
        
        # Reshape advantage and value outputs
        # Proper advantage scaling
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        
        # Scale the outputs to match reward range
        value = value * 10.0  # Scale up to match reward magnitude
        advantage = advantage * 10.0  # Scale advantages to half of value range
        
        q_values = value + advantage
        
        if training:
            self.train()
        return q_values
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        batch_states = {
            "features": torch.FloatTensor(np.stack([s["features"] for s in states])),
            "depth": torch.FloatTensor(np.stack([s["depth"] for s in states]))
        }
        batch_next_states = {
            "features": torch.FloatTensor(np.stack([s["features"] for s in next_states])),
            "depth": torch.FloatTensor(np.stack([s["depth"] for s in next_states]))
        }
        return (batch_states, torch.LongTensor(actions),
                torch.FloatTensor(rewards), batch_next_states,
                torch.FloatTensor(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

class ImprovedDualDQNAgent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float = 3e-4,  # Increased from 1e-4
                 gamma: float = 0.99,
                 tau: float = 0.005,  # Changed from 0.995 to 0.005 (faster updates)
                 epsilon_start: float = 1.0,
                 epsilon_final: float = 0.05,  # Increased from 0.01
                 epsilon_decay: float = 0.995,  # Faster decay
                 buffer_size: int = 500000,
                 batch_size: int = 256,  # Increased from 128
                 update_freq: int = 4): # How often to perform soft updates
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.current_network = DualDQNNetwork(
            env.observation_space, 
            env.action_space.n
        ).to(self.device)
        
        self.target_network = DualDQNNetwork(
            env.observation_space, 
            env.action_space.n
        ).to(self.device)
        
        # Initialize target network with current network's parameters
        self.target_network.load_state_dict(self.current_network.state_dict())
        
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.AdamW(
            self.current_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            amsgrad=True  # Enable AMSGrad variant
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Agent parameters
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_steps = 0
        self.episode_rewards = []
        self.avg_losses = []
        self.avg_q_values = []
    
    def soft_update(self):
        """Perform soft update of target network parameters."""
        with torch.no_grad():
            for target_param, current_param in zip(
                self.target_network.parameters(),
                self.current_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * target_param.data + 
                    (1 - self.tau) * current_param.data
                )
    
    def update_network(self) -> Tuple[float, float]:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = [
            x.to(self.device) if isinstance(x, torch.Tensor) 
            else {k: v.to(self.device) for k, v in x.items()}
            for x in self.replay_buffer.sample(self.batch_size)
        ]

        # Scale rewards to match network output range
        rewards = rewards / 100.0  # Scale down rewards to [-1, 1.5] range

        # Get current Q values
        current_q_values = self.current_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # DDQN: Use current network to select actions
            next_actions = self.current_network(next_states).argmax(dim=1, keepdim=True)
            
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(1, next_actions)
            
            # Compute target Q values with proper scaling
            target_q_values = rewards.unsqueeze(1) + \
                            (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Use Huber loss for better stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), max_norm=1.0)  # Reduced from 10
        self.optimizer.step()

        return loss.item(), current_q_values.mean().item()
    
    def train(self, num_episodes: int, max_steps: int = 1000) -> None:
        """Train the agent for a specified number of episodes."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_q = 0
            updates = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update networks
                if len(self.replay_buffer) >= self.batch_size:
                    loss, avg_q = self.update_network()
                    episode_loss += loss
                    episode_q += avg_q
                    updates += 1
                    
                    # Perform soft update at specified frequency
                    if self.training_steps % self.update_freq == 0:
                        self.soft_update()
                
                state = next_state
                episode_reward += reward
                self.training_steps += 1
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(self.epsilon_final, 
                             self.epsilon * self.epsilon_decay)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            if updates > 0:
                self.avg_losses.append(episode_loss / updates)
                self.avg_q_values.append(episode_q / updates)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_loss = np.mean(self.avg_losses[-10:]) if self.avg_losses else 0
                avg_q = np.mean(self.avg_q_values[-10:]) if self.avg_q_values else 0
                
                print(f"Episode {episode + 1}")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Avg Loss: {avg_loss:.4f}")
                print(f"Avg Q-Value: {avg_q:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print("--------------------")
                
                # Update learning rate based on average reward
                self.scheduler.step(avg_reward)
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"drone_dualdqn_episode_{episode + 1}.pth")
    
    def select_action(self, state: Dict) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = {
                    "features": torch.FloatTensor(state["features"]).unsqueeze(0).to(self.device),
                    "depth": torch.FloatTensor(state["depth"]).unsqueeze(0).to(self.device)
                }
                q_values = self.current_network(state_tensor)
                return q_values.max(1)[1].item()
        return random.randrange(self.env.action_space.n)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'current_network': self.current_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'avg_losses': self.avg_losses,
            'avg_q_values': self.avg_q_values
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.current_network.load_state_dict(checkpoint['current_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint['episode_rewards']
        self.avg_losses = checkpoint['avg_losses']
        self.avg_q_values = checkpoint['avg_q_values']

import os
def main():
    """Main training function."""
    env = DroneEnv()
    agent = ImprovedDualDQNAgent(env)
    if os.path.exists("interrupted_drone_dualdqn.pth"):
        agent.load_model("interrupted_drone_dualdqn.pth")
        print("FineTuning!!")
    try:
        agent.train(num_episodes=100000)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save_model("interrupted_drone_dualdqn.pth")
    finally:
        env.close()

if __name__ == "__main__":
    main()
