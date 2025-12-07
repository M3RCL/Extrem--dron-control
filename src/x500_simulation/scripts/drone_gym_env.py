#!/usr/bin/env python3
"""
Improved Drone Gym Environment - GPS Based with Better Reset Logic

Uses real sensors (no ground truth odometry):
- GPS for position (noisy)
- IMU for orientation and angular velocity
- Barometer for altitude estimation
- Sensor fusion for better state estimation

Better episode management:
- Proper collision detection
- Distance boundary checking
- Gazebo reset integration
- State validation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, NavSatFix, FluidPressure, MagneticField
from nav_msgs.msg import Odometry  # Only for debugging comparison
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from ros_gz_interfaces.srv import SetEntityPose
from threading import Thread, Lock
import time
from scipy.spatial.transform import Rotation as R


class SensorFusion:
    """Simple sensor fusion for state estimation"""
    
    def __init__(self):
        # GPS origin (set on first GPS fix)
        self.gps_origin = None
        self.gps_origin_set = False
        
        # State estimates
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0., 0., 0., 1.])  # quaternion
        self.angular_velocity = np.zeros(3)
        
        # Barometer
        self.altitude_baro = 0.0
        self.pressure_sea_level = 101325.0  # Pa
        
        # Complementary filter weights
        self.alpha_altitude = 0.98  # Trust accelerometer more for short term
        
        # Time tracking
        self.last_update_time = time.time()
    
    def set_gps_origin(self, lat, lon, alt):
        """Set GPS origin for local coordinate conversion"""
        self.gps_origin = (lat, lon, alt)
        self.gps_origin_set = True
    
    def gps_to_local(self, lat, lon, alt):
        """Convert GPS to local ENU coordinates"""
        if not self.gps_origin_set:
            return np.zeros(3)
        
        # Simple flat earth approximation (good for small distances)
        lat0, lon0, alt0 = self.gps_origin
        
        # Meters per degree at equator
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))
        
        # ENU coordinates
        x = (lon - lon0) * meters_per_deg_lon
        y = (lat - lat0) * meters_per_deg_lat
        z = alt - alt0
        
        return np.array([x, y, z])
    
    def pressure_to_altitude(self, pressure_pa):
        """Convert pressure to altitude using barometric formula"""
        altitude = 44330.0 * (1.0 - (pressure_pa / self.pressure_sea_level) ** 0.1903)
        return altitude
    
    def update_gps(self, lat, lon, alt):
        """Update position from GPS"""
        if not self.gps_origin_set:
            self.set_gps_origin(lat, lon, alt)
        
        self.position = self.gps_to_local(lat, lon, alt)
    
    def update_imu(self, orientation_quat, angular_vel, linear_accel):
        """Update orientation and estimate velocity"""
        self.orientation = orientation_quat
        self.angular_velocity = angular_vel
        
        # Simple velocity integration (would need Kalman filter for production)
        dt = time.time() - self.last_update_time
        self.last_update_time = time.time()
        
        if dt < 0.1:  # Sanity check
            # Transform acceleration to world frame
            rot = R.from_quat(orientation_quat)
            accel_world = rot.apply(linear_accel)
            
            # Remove gravity
            accel_world[2] -= 9.81
            
            # Integrate (simple Euler - not perfect but works for RL)
            self.velocity += accel_world * dt
            
            # Apply damping to prevent drift (simulate air resistance)
            self.velocity *= 0.99
    
    def update_barometer(self, pressure):
        """Update altitude from barometer"""
        self.altitude_baro = self.pressure_to_altitude(pressure)
        
        # Fuse with GPS altitude (complementary filter)
        if self.gps_origin_set:
            self.position[2] = self.alpha_altitude * self.position[2] + \
                               (1 - self.alpha_altitude) * self.altitude_baro
    
    def get_state(self):
        """Get full state vector [pos, vel, euler, ang_vel]"""
        # Convert quaternion to euler angles
        rot = R.from_quat(self.orientation)
        euler = rot.as_euler('xyz')
        
        return np.concatenate([
            self.position,      # x, y, z
            self.velocity,      # vx, vy, vz
            euler,              # roll, pitch, yaw
            self.angular_velocity  # wx, wy, wz
        ])


class DroneROSNode(Node):
    """ROS Node for drone sensor communication"""
    
    def __init__(self):
        super().__init__('drone_ros_node')
        
        # Sensor fusion
        self.sensor_fusion = SensorFusion()
        self.state_lock = Lock()
        
        # State tracking
        self.sensors_initialized = False
        self.gps_fix_count = 0
        
        # Ground truth (only for debugging/comparison)
        self.ground_truth_position = np.zeros(3)
        self.use_ground_truth_for_comparison = True
        
        # Publishers
        self.action_pub = self.create_publisher(Float32MultiArray, 'x500/action', 10)
        
        # Subscribers - Real sensors only
        self.gps_sub = self.create_subscription(
            NavSatFix, 'navsat', self.gps_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10
        )
        self.baro_sub = self.create_subscription(
            FluidPressure, 'air_pressure', self.baro_callback, 10
        )
        
        # Ground truth (for debugging only)
        if self.use_ground_truth_for_comparison:
            self.odom_sub = self.create_subscription(
                Odometry, '/model/x500/odometry', self.odom_callback, 10
            )
        
        # Service clients for Gazebo control
        self.reset_world_client = self.create_client(Empty, '/world/drone_world/control/reset')
        self.set_pose_client = self.create_client(Empty, '/world/drone_world/set_pose')
    
    def gps_callback(self, msg):
        """Update position from GPS"""
        with self.state_lock:
            self.sensor_fusion.update_gps(
                msg.latitude,
                msg.longitude,
                msg.altitude
            )
            self.gps_fix_count += 1
            if self.gps_fix_count >= 5:
                self.sensors_initialized = True
    
    def imu_callback(self, msg):
        """Update orientation and angular velocity from IMU"""
        with self.state_lock:
            quat = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])
            
            ang_vel = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            lin_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            
            self.sensor_fusion.update_imu(quat, ang_vel, lin_accel)
    
    def baro_callback(self, msg):
        """Update altitude from barometer"""
        with self.state_lock:
            self.sensor_fusion.update_barometer(msg.fluid_pressure)
    
    def odom_callback(self, msg):
        """Ground truth for debugging comparison"""
        self.ground_truth_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
    
    def get_state(self):
        """Get current state from sensor fusion"""
        with self.state_lock:
            return self.sensor_fusion.get_state()
    
    def publish_action(self, action):
        """Publish action to drone"""
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
    
    def reset_gazebo_world(self, pos=None):
        """Reset Gazebo simulation"""
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('waiting service not available')
            
        
        try:
            robot1_pose = Pose()
            robot1_pose.position.x = pos[0] if pos is not None else 0.0
            robot1_pose.position.y = pos[1] if pos is not None else 0.0
            robot1_pose.position.z = 0.5
            robot1_pose.orientation.w = 1.0  # No rotation
            robot1_pose.orientation.x = 0.0
            robot1_pose.orientation.y = 0.0
            robot1_pose.orientation.z = 0.0
            request = SetEntityPose.Request()
            entity_name = 'x500'
            request.entity.name = entity_name
            request.entity.type = 1
            request.pose = robot1_pose
            future = self.set_pose_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() is not None:
                if future.result().success:
                    self.get_logger().debug(f"Successfully reset {entity_name}")
                    return True
                else:
                    self.get_logger().error(f"Failed to reset {entity_name}: {future.result().message}")
                    return False
            else:
                self.get_logger().error(f"Service call failed for {entity_name}")
                return False
        except Exception as e:
            self.get_logger().error(f'Failed to reset Gazebo: {e}')
    
    def reset_sensors(self):
        """Reset sensor fusion"""
        with self.state_lock:
            self.sensor_fusion = SensorFusion()
            self.sensors_initialized = False
            self.gps_fix_count = 0


class ImprovedDroneEnv(gym.Env):
    """
    Improved Drone Environment with GPS-based positioning and better reset logic
    
    Features:
    - Uses GPS + IMU + Barometer (no ground truth odometry)
    - Proper collision detection
    - Distance boundary enforcement
    - Gazebo reset integration
    - Better reward shaping
    - Episode statistics tracking
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 target_position=None,
                 max_steps=1000,
                 max_distance_from_target=20.0,
                 min_altitude=0.1,
                 max_altitude=10.0,
                 max_tilt_angle=60.0):  # degrees
        
        super().__init__()
        
        # Initialize ROS if not already initialized
        if not rclpy.ok():
            rclpy.init()
        
        # Create ROS node (composition, not inheritance!)
        self.ros_node = DroneROSNode()
        
        # Episode parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_number = 0
        
        # Safety boundaries
        self.max_distance_from_target = max_distance_from_target
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.max_tilt_angle = np.radians(max_tilt_angle)
        # Target
        self.target_position = target_position if target_position is not None else np.array([5., 0., 2.])
        
        # Episode statistics
        self.episode_stats = {
            'termination_reason': None,
            'max_altitude': 0.0,
            'min_altitude': float('inf'),
            'max_distance': 0.0,
            'collision_detected': False
        }
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0., -1., -1., -1.]),
            high=np.array([1., 1., 1., 1.]),
            dtype=np.float32
        )
        
        # State: [pos(3), vel(3), euler(3), ang_vel(3), target(3)] = 15D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        
        # ROS2 spin thread
        self.ros_thread = Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        
        # Wait for sensors
        self.ros_node.get_logger().info('Waiting for sensor data...')
        time.sleep(2.0)
        
        self.ros_node.get_logger().info('Improved Drone Environment initialized (GPS-based)')
    
    def ros_spin(self):
        """Spin ROS2 in separate thread"""
        while rclpy.ok():
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)
    
    def get_obs(self):
        """Get current observation from sensor fusion"""
        state = self.ros_node.get_state()
        
        # Concatenate with target position
        obs = np.concatenate([state, self.target_position])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment with proper Gazebo reset"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_number += 1
        
        # Reset episode statistics
        self.episode_stats = {
            'termination_reason': None,
            'max_altitude': 0.0,
            'min_altitude': float('inf'),
            'max_distance': 0.0,
            'collision_detected': False
        }
        
        # Randomize target if specified
        if options and 'target_position' in options:
            self.target_position = options['target_position']
        elif options and 'randomize_target' in options and options['randomize_target']:
            # Random target in reasonable range
            self.target_position = np.random.uniform(
                [-5, -5, 1],
                [5, 5, 3]
            )
        offset = np.random.uniform(-10,10,size=3)
        spawn_pos = self.target_position + offset
        spawn_pos[2] = max(spawn_pos[2], 0.2)
        # Reset Gazebo world (respawn drone)
        self.ros_node.reset_gazebo_world(spawn_pos)
        
        # Reset sensor fusion
        self.ros_node.reset_sensors()
        
        # Wait for sensors to reinitialize
        timeout = 5.0
        start_time = time.time()
        while not self.ros_node.sensors_initialized and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.ros_node.sensors_initialized:
            self.ros_node.get_logger().warn('Sensors not initialized after reset!')
        
        obs = self.get_obs()
        info = {'episode': self.episode_number}
        
        return obs, info
    
    def reset_gazebo_world(self):
        """Reset Gazebo simulation (deprecated, use ros_node method)"""
        self.ros_node.reset_gazebo_world()
    
    def step(self, action):
        """Execute action and return next state"""
        # Publish action
        self.ros_node.publish_action(action)
        
        # Wait for simulation step
        time.sleep(0.01)
        
        # Get observation
        obs = self.get_obs()
        state = obs[:12]  # Extract state (without target)
        
        # Update episode statistics
        altitude = state[2]
        self.episode_stats['max_altitude'] = max(self.episode_stats['max_altitude'], altitude)
        self.episode_stats['min_altitude'] = min(self.episode_stats['min_altitude'], altitude)
        
        distance = np.linalg.norm(state[:3] - self.target_position)
        self.episode_stats['max_distance'] = max(self.episode_stats['max_distance'], distance)
        
        # Compute reward
        reward = self.compute_reward(state)
        
        # Check termination conditions
        self.current_step += 1
        terminated, termination_reason = self.check_termination(state)
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            self.episode_stats['termination_reason'] = termination_reason
        
        # Build info dict
        info = {
            'position': state[:3].copy(),
            'velocity': state[3:6].copy(),
            'euler_angles': state[6:9].copy(),
            'distance_to_target': distance,
            'altitude': altitude,
            'episode_step': self.current_step,
            'termination_reason': termination_reason if terminated else None,
            'episode_stats': self.episode_stats.copy() if (terminated or truncated) else {}
        }
        
        # Add GPS error for analysis (if debugging enabled)
        if self.ros_node.use_ground_truth_for_comparison:
            gps_error = np.linalg.norm(state[:3] - self.ros_node.ground_truth_position)
            info['gps_position_error'] = gps_error
        
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, state):
        """Improved reward function with better shaping"""
        position = state[:3]
        velocity = state[3:6]
        euler = state[6:9]  # roll, pitch, yaw
        
        # 1. Distance reward (primary objective)
        distance = np.linalg.norm(position - self.target_position)
        distance_reward = -distance
        
        # 2. Progress reward (reward getting closer)
        # Would need to track previous distance for this
        
        # 3. Altitude maintenance (penalize being too low or high)
        altitude_error = 0.0
        if position[2] < self.min_altitude + 0.5:
            altitude_error = (self.min_altitude + 0.5 - position[2]) * 5.0
        elif position[2] > self.max_altitude - 0.5:
            altitude_error = (position[2] - self.max_altitude + 0.5) * 5.0
        
        # 4. Velocity penalty (prefer controlled movement)
        velocity_penalty = -0.1 * np.linalg.norm(velocity)
        
        # 5. Orientation penalty (prefer level flight)
        roll_pitch_penalty = -0.5 * (abs(euler[0]) + abs(euler[1]))
        
        # 6. Success bonus (reached target)
        success_bonus = 0.0
        if distance < 0.5:
            success_bonus = 100.0
        
        # 7. Crash penalty (handled in termination)
        
        # Total reward
        total_reward = (
            distance_reward +
            velocity_penalty +
            roll_pitch_penalty -
            altitude_error +
            success_bonus
        )
        
        return total_reward
    
    def check_termination(self, state):
        """
        Check if episode should terminate
        
        Returns:
            (terminated, reason)
        """
        position = state[:3]
        velocity = state[3:6]
        euler = state[6:9]
        
        # 1. Ground collision
        if position[2] < self.min_altitude:
            self.episode_stats['collision_detected'] = True
            return True, 'ground_collision'
        
        # 2. Flew too high
        if position[2] > self.max_altitude:
            return True, 'altitude_limit'
        
        # 3. Too far from target (flew away)
        distance = np.linalg.norm(position - self.target_position)
        if distance > self.max_distance_from_target:
            return True, 'distance_limit'
        
        # 4. Extreme tilt (crash/flip)
        if abs(euler[0]) > self.max_tilt_angle or abs(euler[1]) > self.max_tilt_angle:
            self.episode_stats['collision_detected'] = True
            return True, 'extreme_tilt'
        
        # 5. Excessive velocity (crash)
        speed = np.linalg.norm(velocity)
        if speed > 10.0:  # 10 m/s max
            return True, 'excessive_velocity'
        
        # 6. Success (reached target)
        if distance < 0.5:
            return True, 'success'
        
        return False, None
    
    def close(self):
        """Clean up resources"""
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Example usage
if __name__ == '__main__':
    env = ImprovedDroneEnv(
        target_position=np.array([5., 0., 2.]),
        max_distance_from_target=15.0
    )
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Episode: {info['episode']}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: distance={info['distance_to_target']:.2f}m, "
                  f"altitude={info['altitude']:.2f}m, reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {info['termination_reason']}")
            print(f"Episode stats: {info['episode_stats']}")
            break
    
    env.close()