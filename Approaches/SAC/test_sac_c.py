import airsim
import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import math
import os


#############################################
# Define a custom Gym Environment for Drone #
############################################## Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Define a custom Gym Environment for Drone #
#############################################
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # Initialize AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Simulation parameters
        self.dt = 0.1         # Duration per step (seconds)
        self.step_count = 0
        # Throttle range parameters
        self.max_throttle = 75.0  # Upper bound for throttle
        self.min_throttle = 0.0  # Lower bound for throttle
        
        # Define maximum values for other controls (in physical units)
        self.max_roll = math.pi/2    # radians (~90 degrees)
        self.max_pitch = math.pi/2   # radians (~90 degrees)
        self.max_yaw_rate = 1.0 # radians per second
        
        # Waypoint-based target parameters
        self.target_param = 0.0      # Parameter to compute the current waypoint along the helix
        self.target_increment = 2.0  # How much to advance when a waypoint is reached
        self.target_threshold = 2.0  # Distance (meters) to consider a waypoint reached
        self.target_bonus = 250.0     # Bonus reward for reaching a waypoint
        self.current_target = None
        self.next_target = None
        # Curriculum factor (from 0 to 1); extra reward terms will be scaled by this factor.
        self.curriculum_factor = 0.0
        
        # Define action space: [roll, pitch, yaw_rate, throttle] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation space: state vector of dimension 15:
        # [pos.x, pos.y, pos.z, euler0, euler1, euler2,
        #  rel_pos_body[0], rel_pos_body[1], rel_pos_body[2],
        #  lin_vel.x, lin_vel.y, lin_vel.z, ang_vel.x, ang_vel.y, ang_vel.z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # Variables for tracking progress
        self.prev_speed = None
        self.prev_x = None
        self.np_random = None
        self.seed()
        #self.reset()
    
    def set_curriculum_factor(self, factor):
        self.curriculum_factor = factor
        
    def seed(self, seed=1):
        super().reset(seed=seed)
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return [seed]
    def draw_target_marker(self, target):
        """
        Draw a marker at the target point in Unreal using simPlotPoints.
        """
        point = airsim.Vector3r(target[0], target[1], target[2])
        color = [1.0, 0.0, 0.0, 1.0]  # Red, fully opaque
        size = 10.0                   # Marker size (pixels)
        duration = 0.1                # Duration for which the marker is drawn
        is_persistent = False         # Not persistent; re-drawn each step
        self.client.simPlotPoints([point], color, size, duration, is_persistent)

    def helix_target(self, param):
        """
        Compute a waypoint on a forward-spreading helix.
        Here the helix is defined so that:
          target_x = R*cos(param) - 15,
          target_y = R*sin(param),
          target_z = - (forward_rate * param) - 55.
        Adjust R and forward_rate as desired.
        """
        R = 15.0           # Helix radius
        forward_rate = 4.0  # Controls how fast the helix moves forward (in z)
        target_x = R * math.cos(param) - 15
        target_y = R * math.sin(param)
        target_z = -forward_rate * param - 75
        return np.array([target_x, target_y, target_z])
    def random_target(self, reference_position):
        # Define the allowed offset ranges
        x_offset = np.random.uniform(-10, 10)
        y_offset = np.random.uniform(-10, 10)
        z_offset = np.random.uniform(-5, 5)
        
        # You might also ensure that the overall distance is at least a minimum value:
        offset = np.array([x_offset, y_offset, z_offset])
        min_dist = np.random.uniform(1, 10)
        while np.linalg.norm(offset) < min_dist:
            x_offset = np.random.uniform(-10, 10)
            y_offset = np.random.uniform(-10, 10)
            z_offset = np.random.uniform(-5, 5)
            offset = np.array([x_offset, y_offset, z_offset])
        
        return reference_position + offset


    def transform_to_body_frame(self, point, quat):
        """
        Transform a point from world frame to the drone's body frame.
        The quaternion is assumed to be (w, x, y, z).
        """
        w, x, y, z = quat
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
        ])
        return rotation_matrix.T @ point

    def _get_obs(self):
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orientation = state.kinematics_estimated.orientation
            lin_vel = state.kinematics_estimated.linear_velocity
            ang_vel = state.kinematics_estimated.angular_velocity

            # Get the drone's Euler angles (roll, pitch, yaw)
            euler = airsim.to_eularian_angles(orientation)  # (roll, pitch, yaw)
            
            # Compute the current target position using the waypoint-based helix
            goal = self.current_target
            
            # Compute desired yaw as the angle from the drone to the target (in the x-y plane)
            desired_yaw = math.atan2(goal[1] - pos.y_val, goal[0] - pos.x_val)
            yaw_error = euler[2] - desired_yaw
            yaw_error = (yaw_error + math.pi) % (2*math.pi) - math.pi  # Normalize to [-pi, pi]
            
            # Compute desired pitch from the drone to the target.
            horizontal_dist = math.sqrt((goal[0] - pos.x_val)**2 + (goal[1] - pos.y_val)**2)
            desired_pitch = math.atan2(-(goal[2] - pos.z_val), horizontal_dist)
            pitch_error = euler[1] #- desired_pitch
            
            # Assume desired roll is zero.
            roll_error = euler[0]
            
            # Instead of raw orientation, we now use the orientation error.
            # Also, compute the relative position to the target (in the body frame).
            rel_pos = np.array([pos.x_val - goal[0], pos.y_val - goal[1], pos.z_val - goal[2]])
            rel_pos_body = self.transform_to_body_frame(
                                rel_pos,
                                np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
                            )
            
            # Build the state vector.
            # Here, we replace the raw Euler angles with the orientation errors.
            obs = np.array([
                #pos.x_val, pos.y_val, pos.z_val,
                roll_error, pitch_error, yaw_error,
                rel_pos_body[0], rel_pos_body[1], rel_pos_body[2],
                lin_vel.x_val, lin_vel.y_val, lin_vel.z_val,
                ang_vel.x_val, ang_vel.y_val, ang_vel.z_val
            ], dtype=np.float32)
            return obs

    def _calculate_reward(self, obs):
        """
        Compute the composite reward with improved d_dot calculation.
        """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        real_state = self.client.getMultirotorState()
        lin_vel = real_state.kinematics_estimated.linear_velocity
        velocity = np.array([lin_vel.x_val, lin_vel.y_val, lin_vel.z_val])
        target = self.current_target
        alt_er = target[-1] - pos[-1]
        next_target = self.next_target
        current_distance = np.linalg.norm(target - pos)
        d = current_distance
        # Exponential distance reward (better gradient)
        distance_reward = math.exp(-0.02* current_distance**2)
        #distance_reward = -current_distance**2  # Quadratic distance reward
        # Direction from position to target
        direction = target - pos
        
        # Progress reward: reduction in distance from previous step
        
        try:
            lin_acc = real_state.kinematics_estimated.linear_acceleration
            acceleration = np.array([lin_acc.x_val, lin_acc.y_val, lin_acc.z_val])
        except AttributeError:
            # If acceleration is not directly available, you might approximate it using finite differences of velocity.
            acceleration = np.zeros_like(velocity)
        # Calculate d_dot: projection of velocity onto direction to target
        # This measures how directly the drone is moving toward the target
        if np.linalg.norm(direction) > 1e-6:
            direction_norm = direction / np.linalg.norm(direction)
            #z_norm = (alt_er**2)
            d_dot = np.dot( velocity, direction_norm)
            #z_dot = -0.01*(z_norm)
            numerator = np.dot(target - pos, acceleration) - np.linalg.norm(velocity)**2 + d_dot**2
            d_dot_dot = numerator / d
        else:
            d_dot = 0.0
            z_dot = 0.0
            d_dot_dot = 0.0
        # For acceleration, if available, try to get it from the simulator:
        
        if self.prev_speed is not None:
            acc_reward = (d_dot - self.prev_speed)/self.dt
        else:
            acc_reward = 0.0    
        # Improved d_dot reward - positive when moving toward target, negative when moving away
        lamda = 1.0  # Increased weight for this component
        #d_dot_clipped = np.clip(d_dot, -5.0, 5.0)
        d_dot_reward =  lamda * (d_dot)
        lada_acc = 0.5
        d_dot_reward = d_dot_reward + lada_acc *  d_dot_dot
        #print("d_dot_reward", d_dot_reward)
        #print(z_dot)
        # Alignment reward: how well the drone's velocity is aligned with the vector toward the target
        alignment_reward = 0.0
        if np.linalg.norm(velocity) > 1e-6:
            direction_to_target = direction / np.linalg.norm(direction)
            velocity_norm = velocity / np.linalg.norm(velocity)
            alignment_reward = np.dot(velocity_norm, direction_to_target)
        
        # Calculate the shortest distance to the path segment
        path_vector = next_target - target
        if np.linalg.norm(path_vector) > 1e-6:
            path_vector_norm = path_vector / np.linalg.norm(path_vector)
            pos_to_target = pos - target
            # Project position onto path vector
            proj_length = np.dot(pos_to_target, path_vector_norm)
            # Calculate perpendicular distance to path
            perp_vector = pos_to_target - proj_length * path_vector_norm
            perp_distance = np.linalg.norm(perp_vector)
            # Reward for staying close to the path
            path_following_reward = math.exp(-0.5 * perp_distance)
        else:
            path_following_reward = 0.0
        
        # Base reward is always applied
        base_reward = distance_reward + d_dot_reward - 0.5 # Small constant cost per time step
        # 0.01 * alignment_reward +
        # Additional reward components, scaled by curriculum factor
        additional_reward = (
            
            0.2 * path_following_reward +
            0.1 * alignment_reward
        )
        
        reward = base_reward #+ self.curriculum_factor * additional_reward
        
        self.prev_speed = d_dot
        return reward

    def _check_done(self, obs):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        target = self.current_target
        if np.linalg.norm(target - pos) > 50:
            return True
        if self.step_count >= 1000:
            return True
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        return False

    def step(self, action):
        self.step_count += 1
        
        roll = float(action[0]) * self.max_roll
        pitch = float(action[1]) * self.max_pitch
        yaw_rate = float(action[2]) * self.max_yaw_rate
        base_throttle = (float(action[3]) + 1.0) / 2.0
        
        acrobatic_factor = max(abs(action[0]), abs(action[1]))
        dynamic_gain = 1 + (acrobatic_factor - 0.5) * 2.0 if acrobatic_factor > 0.5 else 1.0
        norm_throttle = (float(action[3]) + 1.0) / 2.0
        throttle = norm_throttle * (self.max_throttle ) 
        throttle = throttle #* dynamic_gain # dynamic_gain can be applied if desired
        
        # Command the drone.
        self.client.moveByRollPitchYawrateThrottleAsync(roll, pitch, yaw_rate, throttle, self.dt).join()
        
        current_target = self.current_target
        self.draw_target_marker(current_target)
        obs = self._get_obs()
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        current_pos = pos
        reward = self._calculate_reward(obs)
        
        # If the drone strays too far from current target, impose an extra penalty.
        if np.linalg.norm(current_pos - current_target) > 50:
            reward -= 50
            
        done = self._check_done(obs)
        
        # Check if the current waypoint is reached.
        distance_to_target = np.linalg.norm(current_pos - current_target)
        if distance_to_target < self.target_threshold:
            print("Waypoint reached!")
            reward += self.target_bonus
            self.current_target = self.next_target
            self.next_target = self.random_target(self.current_target)  # Advance to next waypoint.
            obs = self._get_obs()  # Update observation with new target.
        
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 1000
        
        self.prev_x = obs[0]
        info = {}
        return obs, reward, done, info

    def reset(self, *, seed=1, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.target_param = 0.0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        delta = 5.0
        # Create a random offset in each dimension
        offset = np.random.uniform(-delta, delta, size=3)
        spawn_pos = [55, -50 , -75] + offset
        fixed_pose = airsim.Pose(airsim.Vector3r(spawn_pos[0], spawn_pos[1], spawn_pos[2]), airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(fixed_pose, True)
        self.client.takeoffAsync().join()
        self.current_target = self.random_target(spawn_pos)
        self.next_target = self.random_target(self.current_target)
        obs = self._get_obs()
        pos = obs[:3]
        
        self.prev_speed = None
        self.prev_x = obs[0]
        return obs
#############################################
# SAC Actor, Critic, Replay Buffer, and SAC #
#############################################
class CascadedActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(CascadedActor, self).__init__()
        
        # Based on the provided state vector:
        # [pos.x, pos.y, pos.z, euler[0], euler[1], euler[2], 
        #  rel_pos_body[0], rel_pos_body[1], rel_pos_body[2],
        #  lin_vel.x, lin_vel.y, lin_vel.z, 
        #  ang_vel.x, ang_vel.y, ang_vel.z]
        
        # Position components: indices 0-8 (positions, orientation, relative goal)
        self.position_indices = list(range(0, 6))
        self.position_dim = len(self.position_indices)
        
        # Velocity components: indices 9-14 (linear and angular velocities)
        self.velocity_indices = list(range(6, 12))
        self.velocity_dim = len(self.velocity_indices)
        
        self.action_dim = action_dim
        
        # First part: Position Controller
        self.pos_l = nn.Linear(self.position_dim, 256)
        self.pos_l1 = nn.Linear(256, 128)
        self.pos_l2 = nn.Linear(128, 64)
        
        # Output desired velocities (same dimension as actual velocities)
        self.desired_vel = nn.Linear(64, self.velocity_dim)
        
        # Second part: Velocity Controller
        self.vel_input_dim = self.velocity_dim + 64  # desired vel + actual vel + position features
        self.vel_l1 = nn.Linear(self.vel_input_dim, 256)
        self.vel_l2 = nn.Linear(256+self.velocity_dim, 128)
        self.vel_l3 = nn.Linear(128, 64)
        
        # Final action output
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        # Extract position and velocity components
        pos_state = state[:, self.position_indices]
        vel_state = state[:, self.velocity_indices]
        
        # Position controller - outputs desired velocities
        p = torch.relu(self.pos_l(pos_state))
        p = torch.relu(self.pos_l1(p))
        p = torch.relu(self.pos_l2(p))
        
        # Keep position features for the velocity controller
        pos_features = p
        
        # Calculate desired velocities
        desired_vel = self.desired_vel(p)
        
        # Concatenate desired velocities, actual velocities, and position features
        combined = torch.cat([ vel_state, pos_features], dim=1)
        
        # Velocity controller - outputs actions
        v = torch.relu(self.vel_l1(combined))
        v = torch.relu(self.vel_l2(torch.cat([v, desired_vel], dim=1)))
        v = torch.relu(self.vel_l3(v))
        
        mean = self.mean(v)
        log_std = self.log_std(v)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std

    def sample(self, state, deterministic = False):
        mean, log_std = self.forward(state)
        if deterministic:
            # Use mean action directly (no noise)
            y_t = torch.tanh(mean)              # Squash mean to [-1, 1]
            action = y_t * self.max_action      # Scale to action space
            return action, None                 # Log_prob not needed for testing
        else:
            std = torch.exp(log_std)
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.max_action
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return action, log_prob
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Define indices for the cascaded structure (matching the actor)
        # Our state vector is assumed to be 15-dimensional:
        # [pos.x, pos.y, pos.z, euler0, euler1, euler2,
        #  rel_pos_body[0], rel_pos_body[1], rel_pos_body[2],
        #  lin_vel.x, lin_vel.y, lin_vel.z, ang_vel.x, ang_vel.y, ang_vel.z]
        # We take the first 9 as "position-related" features and 9-15 as "velocity-related"
        self.position_indices = list(range(0, 6))
        self.velocity_indices = list(range(6, 12))
        pos_dim = len(self.position_indices)   # 9
        vel_dim = len(self.velocity_indices)   # 6

        # --- Q1 Network ---
        # Position branch: compute intermediate features from position part
        self.pos_branch1 = nn.Sequential(
            nn.Linear(pos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Desired velocity from Q1 position branch (same dimension as vel_dim)
        self.desired_vel1 = nn.Linear(64, vel_dim)
        # Combined branch: input is [desired_vel1, actual velocity, pos_features, action]
        combined_dim = vel_dim  + 64 + action_dim
        self.combined_branch1 = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),)
        self.combined_branch11 = nn.Sequential(
            nn.Linear(256+ vel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
            
        )

        # --- Q2 Network ---
        self.pos_branch2 = nn.Sequential(
            nn.Linear(pos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.desired_vel2 = nn.Linear(64, vel_dim)
        self.combined_branch2 = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU())
        self.combined_branch22 = nn.Sequential(
            nn.Linear(256+ vel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # Extract position and velocity parts from the state.
        pos_state = state[:, self.position_indices]  # shape: [batch, 9]
        vel_state = state[:, self.velocity_indices]    # shape: [batch, 6]
        
        # Q1 forward
        pos_feat1 = self.pos_branch1(pos_state)         # shape: [batch, 64]
        desired_vel1 = self.desired_vel1(pos_feat1)       # shape: [batch, 6]
        combined1 = torch.cat([vel_state, pos_feat1, action], dim=1)
        q1 = self.combined_branch1(combined1)
        q1 = self.combined_branch11(torch.cat([q1, desired_vel1], dim=1))# shape: [batch, 1]
        
        # Q2 forward
        pos_feat2 = self.pos_branch2(pos_state)         # shape: [batch, 64]
        desired_vel2 = self.desired_vel2(pos_feat2)       # shape: [batch, 6]
        combined2 = torch.cat([ vel_state, pos_feat2, action], dim=1)
        q2 = self.combined_branch2(combined2)     
        q2 = self.combined_branch22(torch.cat([q2, desired_vel2], dim=1))# shape: [batch, 1]
        
        return q1, q2


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.max_size = int(max_size)
        self.storage = []
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return (np.array(batch_states),
                np.array(batch_actions),
                np.array(batch_rewards).reshape(-1, 1),
                np.array(batch_next_states),
                np.array(batch_dones).reshape(-1, 1))

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = CascadedActor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initial learning rates
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=5000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=5000, gamma=0.9)
        
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        
        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)  # -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = torch.exp(self.log_alpha).item()
        
        # Gradient clipping
        self.max_grad_norm = 1500
        
        # Training step counter
        self.train_steps = 0

    def select_action(self, state, deterministic=True):
        
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _ = self.actor.sample(state, deterministic = True)
            return action.cpu().data.numpy().flatten()

    def save(self, filename):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'train_steps': self.train_steps
        }
        torch.save(checkpoint, filename)
        
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load alpha-related parameters if they exist in the checkpoint
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = torch.exp(self.log_alpha).item()
            
        if 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
        # Load scheduler states if they exist
        if 'actor_scheduler_state_dict' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            
        if 'critic_scheduler_state_dict' in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']
        

        print(f"Model loaded from {filename}")

    def train(self, replay_buffer, batch_size=256):
        self.train_steps += 1
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.discount * target_q
            
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update actor
        action_new, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Apply gradient clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update alpha (automatic entropy tuning)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha).item()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Step the learning rate schedulers
        if self.train_steps % 100 == 0:  # Every 100 training steps
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
def test_agent(model_path, num_episodes=5, max_steps=1000):
        """Test a saved agent in the environment."""
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = DroneEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        # Create agent and load model
        agent = SAC(state_dim, action_dim, max_action)
        agent.load(model_path)
        agent.actor.eval()  # Set to evaluation mode
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for t in range(max_steps):
                # Always use deterministic actions during testing
                action = agent.select_action(state, deterministic=True)
                
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                    
            print(f"Test Episode {episode}: Reward = {episode_reward}")

# Call the test function when needed
if __name__ == "__main__":
    # After training or separately
    test_agent("drone_SAC_C_last.pth", num_episodes=3)