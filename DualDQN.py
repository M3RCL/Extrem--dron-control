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
    
    def __init__(self, waypoint_list=None, max_episode_steps=350, client_id=0, render_mode='human'):
        super(DroneEnv, self).__init__()
        
        # Use different client IDs for parallel environments
        self.client_id = client_id
        self.render_mode = render_mode
        self.thresh_dist = 35.0
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
        
        self.step_length = 50.0
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
                low=np.array([-50, -math.pi, -50, -25, -25, -25, -10, -10, -10], dtype=np.float32),
                high=np.array([ 50,  math.pi,  50,  25,  25,  25,  10,  10,  10], dtype=np.float32)
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

        if waypoint_list is None:
            self.waypoints = [
                np.array([-5.55265, -31.9786, -19.0225]),
                np.array([-48.59735, -63.3286, -60.07256]),
                np.array([-193.5974, -55.0786, -46.32256]),
                np.array([-369.2474, 35.32137, -62.5725]),
                np.array([-241.3474, 143.6714, -32.07256]),
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
        self.reward = 0
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.successful_waypoints = 0
        
        self.np_random = None
        self.seed()     
        
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
        try:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            start_pose = airsim.Pose()
            start_pose.position.x_val = -10
            start_pose.position.y_val = -20
            start_pose.position.z_val = -20
            self.client.simSetVehiclePose(start_pose, True)
            time.sleep(0.1)
            self.client.takeoffAsync().join()
            state = self._get_state()
            self.prev_distance = self._distance_to_goal(state["features"])
            self.waypoint_start_step = self.step_count
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
                    1
                ).join()
            except Exception as e:
                print(f"Error applying translation action: {e}")
                return self._get_state(), -100, True, True, {}
        else:
            yaw_rate = 30 if action == 5 else -30
            try:
                self.client.rotateByYawRateAsync(yaw_rate, 1).join()
            except Exception as e:
                print(f"Error applying rotation action: {e}")
                return self._get_state(), -100, True, True, {}

        state = self._get_state()
        reward = self._compute_enhanced_reward(state)
        terminated = False
        truncated = False
        if reward <= -40:
            terminated = True
            print("Way away from point reward", reward)
        if self.client.simGetCollisionInfo().has_collided:
            reward = -100.0
            self.collision_count += 1
            print("Collision detected!")
            terminated = True
        reward = self._shape_reward(reward, state, {})
        
        curr_distance = self._distance_to_goal(state["features"])
        if curr_distance < self.goal_threshold:
            reward += 75.0
            self.successful_waypoints += 1
            print("Waypoint reached!")
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
        pts = self.waypoints
        pose = self.client.simGetVehiclePose()
        current_pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        dist = 1e7
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((current_pos - pts[i]), (current_pos - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1])
            )
        if dist > self.thresh_dist:
            reward = -40
        else:
            reward_dist = math.exp(-self.beta * dist) - 0.5
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed + 0.06 * (self.waypoint_start_step - self.step_count)
        # Small reward for heading alignment
        heading_error = state["features"][1]
        heading_reward = 0.01 * math.cos(heading_error)
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
        pose = self.client.simGetVehiclePose()
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

        dx = goal_x - drone_x
        dy = goal_y - drone_y
        distance = math.sqrt(dx**2 + dy**2)
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
            altitude_error,
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

# Larger DualDQNNetwork architecture (Dropout layers removed)
class DualDQNNetwork(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int):
        super(DualDQNNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
        
        self.feature_net = nn.Sequential(
            nn.Linear(observation_space.spaces["features"].shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        training = self.training
        if not training:
            self.eval()
        depth = observation["depth"].unsqueeze(1)
        cnn_features = self.cnn(depth)
        vector_features = self.feature_net(observation["features"])
        combined = torch.cat((cnn_features, vector_features), dim=1)
        fused_features = self.fusion(combined)
        advantage = self.advantage(fused_features)
        value = self.value(fused_features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
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

class DualDQNAgent:
    def __init__(self, env: gym.Env, learning_rate: float = 3e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_final: float = 0.01, epsilon_decay: float = 0.9995,
                 buffer_size: int = 500000, batch_size: int = 128,
                 target_update_freq: int = 2000):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_network = DualDQNNetwork(env.observation_space, env.action_space.n).to(self.device)
        self.target_network = DualDQNNetwork(env.observation_space, env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.current_network.state_dict())
        self.optimizer = optim.AdamW(self.current_network.parameters(),
                                     lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='max', factor=0.5,
                                                              patience=5, verbose=True)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_steps = 0
        self.episode_rewards = []
        self.avg_losses = []
        self.avg_q_values = []
        
    def select_action(self, state: Dict) -> int:
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = {
                    "features": torch.FloatTensor(state["features"]).unsqueeze(0).to(self.device),
                    "depth": torch.FloatTensor(state["depth"]).unsqueeze(0).to(self.device)
                }
                q_values = self.current_network(state_tensor)
                return q_values.max(1)[1].item()
        return random.randrange(self.env.action_space.n)
    
    def update_network(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = [
            x.to(self.device) if isinstance(x, torch.Tensor) else {k: v.to(self.device) for k, v in x.items()}
            for x in self.replay_buffer.sample(self.batch_size)
        ]
        current_q_values = self.current_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.current_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), 10)
        self.optimizer.step()
        return loss.item()
    
    def train(self, num_episodes: int, max_steps: int = 1000) -> None:
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                loss = self.update_network()
                self.training_steps += 1
                if self.training_steps % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.current_network.state_dict())
                if done:
                    break
            self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
            self.episode_rewards.append(episode_reward)
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            if (episode + 1) % 100 == 0:
                self.save_model(f"drone_dualdqn_episode_{episode + 1}.pth")
    
    def save_model(self, path: str) -> None:
        torch.save({
            'current_network': self.current_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.current_network.load_state_dict(checkpoint['current_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

import os
def main():
    """Main training function."""
    env = DroneEnv()
    agent = DualDQNAgent(env)
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
