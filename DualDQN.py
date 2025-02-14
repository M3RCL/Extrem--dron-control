#!/usr/bin/env python3
"""
Dual DQN Training Script for Aggressive Drone Navigation with Full Depth Image Input

This script modifies the original PPO implementation to use Dual DQN instead.
The observation and action spaces remain the same, but the training approach
uses two Q-networks (current and target) to reduce overestimation bias.
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

# Keep the original DroneEnv class exactly as is
class DroneEnv(gym.Env):
    """Enhanced version of the DroneEnv with Gymnasium compatibility"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, waypoint_list=None, max_episode_steps=350, client_id=0, render_mode='human'):
        super(DroneEnv, self).__init__()
        
        # Use different client IDs for parallel environments
        
        self.client_id = client_id
        self.render_mode = render_mode
        self.thresh_dist = 25.0  # Maximum distance threshold for reward calculation
        self.beta = 1.0  # Distance scaling factor
        self.speed_weight = 0.5  # Weight for speed reward
        self.min_speed = -0.05  # Minimum desired speed
        self.max_speed = 50.0  # Maximum desired speed
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
            np.array([-0.55265, -31.9786, -19.0225]),
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
            start_pose.position.y_val = -20
            start_pose.position.z_val = self.np_random.uniform(-20, -30)
            
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
            reward += 1000.0/self.step_count  # Bonus for reaching waypoint
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
            np.array([-0.55265, -31.9786, -19.0225]),
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

class DualDQNNetwork(nn.Module):
    """
    Enhanced neural network for Dual DQN with larger capacity for complex navigation.
    Fixed to handle single samples during inference.
    """
    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int):
        super(DualDQNNetwork, self).__init__()
        
        # Enhanced CNN for depth image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
        
        # Enhanced MLP for feature vector processing
        self.feature_net = nn.Sequential(
            nn.Linear(observation_space.spaces["features"].shape[0], 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Replace BatchNorm with Dropout
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_flatten + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions)
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Set dropout layers to eval mode during inference
        training = self.training
        if not training:
            self.eval()
            
        # Process depth image
        depth = observation["depth"].unsqueeze(1)  # Add channel dimension
        cnn_features = self.cnn(depth)
        
        # Process feature vector
        vector_features = self.feature_net(observation["features"])
        
        # Combine features
        combined = torch.cat((cnn_features, vector_features), dim=1)
        fused_features = self.fusion(combined)
        
        # Calculate advantage and value streams
        advantage = self.advantage(fused_features)
        value = self.value(fused_features)
        
        # Combine streams using dueling architecture formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Restore training mode if necessary
        if training:
            self.train()
            
        return q_values


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Dict, action: int, reward: float, 
             next_state: Dict, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert states and next_states to appropriate format
        batch_states = {
            "features": torch.FloatTensor(np.stack([s["features"] for s in states])),
            "depth": torch.FloatTensor(np.stack([s["depth"] for s in states]))
        }
        
        batch_next_states = {
            "features": torch.FloatTensor(np.stack([s["features"] for s in next_states])),
            "depth": torch.FloatTensor(np.stack([s["depth"] for s in next_states]))
        }
        
        return (batch_states,
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                batch_next_states,
                torch.FloatTensor(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DualDQNAgent:
    def __init__(self, env: gym.Env, 
                 learning_rate: float = 3e-4,  # Slightly higher learning rate
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_final: float = 0.01,
                 epsilon_decay: float = 0.9995,  # Slower decay
                 buffer_size: int = 500000,  # Larger buffer
                 batch_size: int = 128,      # Larger batch size
                 target_update_freq: int = 2000):  # Less frequent updates
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.current_network = DualDQNNetwork(env.observation_space, 
                                            env.action_space.n).to(self.device)
        self.target_network = DualDQNNetwork(env.observation_space, 
                                           env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.current_network.state_dict())
        
        # Enhanced optimizer with gradient clipping
        self.optimizer = optim.AdamW(  # Using AdamW instead of Adam
            self.current_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Other parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Larger replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Enhanced metrics tracking
        self.training_steps = 0
        self.episode_rewards = []
        self.avg_losses = []
        self.avg_q_values = []
        
    def select_action(self, state: Dict) -> int:
        if random.random() > self.epsilon:
            with torch.no_grad():
                # Convert state to tensor and add batch dimension
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
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            [x.to(self.device) if isinstance(x, torch.Tensor) else 
             {k: v.to(self.device) for k, v in x.items()}
             for x in self.replay_buffer.sample(self.batch_size)]
        
        # Get current Q values
        current_q_values = self.current_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values using target network for value estimation
        with torch.no_grad():
            # Select actions using current network
            next_actions = self.current_network(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + \
                            (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int, max_steps: int = 1000) -> None:
        """Main training loop."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                
                # Update network
                loss = self.update_network()
                self.training_steps += 1
                
                # Update target network
                if self.training_steps % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.current_network.state_dict())
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(self.epsilon_final, 
                             self.epsilon * self.epsilon_decay)
            
            # Log progress
            self.episode_rewards.append(episode_reward)
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"drone_dualdqn_episode_{episode + 1}.pth")
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'current_network': self.current_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path)
        self.current_network.load_state_dict(checkpoint['current_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
import os
def main():
    """Main training function."""
    # Create environment
    env = DroneEnv()
    
    # Create and train agent
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