#!/usr/bin/env python3
"""
Gymnasium Environment with RViz Visualization Support
Publishes training progress and targets to RViz during training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from threading import Thread, Lock
import time


class DroneEnvWithViz(gym.Env, Node):
    """
    Drone Environment with RViz Visualization
    Shows target positions, trajectory predictions, and training progress
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, target_position=None, max_steps=1000, visualize=True):
        rclpy.init()
        Node.__init__(self, 'drone_env_viz')
        
        self.max_steps = max_steps
        self.current_step = 0
        self.target_position = target_position if target_position is not None else np.array([0., 0., 2.])
        self.visualize = visualize
        
        # Episode tracking
        self.episode_number = 0
        self.episode_reward = 0.0
        self.best_distance = float('inf')
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0., -1., -1., -1.]),
            high=np.array([1., 1., 1., 1.]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # State storage
        self.state = np.zeros(12)
        self.state_lock = Lock()
        
        # Publishers
        self.action_pub = self.create_publisher(Float32MultiArray, 'x500/action', 10)
        self.viz_marker_pub = self.create_publisher(MarkerArray, 'training_markers', 10)
        
        # Subscribers
        self.state_sub = self.create_subscription(
            Float32MultiArray, 'x500/state', self.state_callback, 10
        )
        
        # ROS2 spin thread
        self.ros_thread = Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        
        time.sleep(1.0)
        self.get_logger().info('Drone Environment with Visualization initialized')
    
    def ros_spin(self):
        """Spin ROS2 in separate thread"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
    
    def state_callback(self, msg):
        """Update state from ROS2 topic"""
        with self.state_lock:
            self.state = np.array(msg.data, dtype=np.float32)
    
    def get_obs(self):
        """Get current observation"""
        with self.state_lock:
            state_copy = self.state.copy()
        obs = np.concatenate([state_copy, self.target_position])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment and update visualizations"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_number += 1
        self.episode_reward = 0.0
        self.best_distance = float('inf')
        
        # Randomize target position if options provided
        if options and 'target_position' in options:
            self.target_position = options['target_position']
        
        # Publish target marker
        if self.visualize:
            self.publish_target_marker()
            self.publish_episode_info()
        
        time.sleep(0.1)
        obs = self.get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute action and update visualizations"""
        # Publish action
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
        
        time.sleep(0.01)
        
        # Get new observation
        obs = self.get_obs()
        
        # Compute reward
        reward = self.compute_reward()
        self.episode_reward += reward
        
        # Update best distance
        distance = np.linalg.norm(self.state[:3] - self.target_position)
        if distance < self.best_distance:
            self.best_distance = distance
        
        # Check termination
        self.current_step += 1
        terminated = self.is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Visualize progress
        if self.visualize and self.current_step % 10 == 0:
            self.publish_training_progress()
        
        info = {
            'distance_to_target': distance,
            'position': self.state[:3].copy(),
            'velocity': self.state[3:6].copy(),
            'episode_reward': self.episode_reward,
            'best_distance': self.best_distance
        }
        
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self):
        """Compute reward based on current state"""
        position = self.state[:3]
        velocity = self.state[3:6]
        orientation = self.state[6:9]
        
        distance = np.linalg.norm(position - self.target_position)
        distance_reward = -distance
        velocity_penalty = -0.1 * np.linalg.norm(velocity)
        orientation_penalty = -0.5 * (abs(orientation[0]) + abs(orientation[1]))
        altitude_penalty = -10.0 if position[2] < 0.1 else 0.0
        success_bonus = 100.0 if distance < 0.5 else 0.0
        
        total_reward = (distance_reward + velocity_penalty + 
                        orientation_penalty + altitude_penalty + success_bonus)
        return total_reward
    
    def is_terminated(self):
        """Check if episode should terminate"""
        position = self.state[:3]
        
        if position[2] < 0.05:
            return True
        if np.linalg.norm(position) > 50.0:
            return True
        
        distance = np.linalg.norm(position - self.target_position)
        if distance < 0.5:
            return True
        
        return False
    
    def publish_target_marker(self):
        """Publish target position marker"""
        marker_array = MarkerArray()
        
        # Target sphere
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'rl_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.target_position[0])
        marker.pose.position.y = float(self.target_position[1])
        marker.pose.position.z = float(self.target_position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8)
        marker.lifetime.sec = 0
        
        marker_array.markers.append(marker)
        
        # Target arrow pointing down
        arrow = Marker()
        arrow.header.frame_id = 'world'
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'rl_target_arrow'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        
        start = Point()
        start.x = float(self.target_position[0])
        start.y = float(self.target_position[1])
        start.z = float(self.target_position[2]) + 2.0
        
        end = Point()
        end.x = float(self.target_position[0])
        end.y = float(self.target_position[1])
        end.z = float(self.target_position[2])
        
        arrow.points = [start, end]
        arrow.scale.x = 0.1
        arrow.scale.y = 0.2
        
        arrow.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        
        marker_array.markers.append(arrow)
        
        self.viz_marker_pub.publish(marker_array)
    
    def publish_training_progress(self):
        """Publish training progress indicators"""
        marker_array = MarkerArray()
        
        # Progress line from drone to target
        line = Marker()
        line.header.frame_id = 'world'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'progress_line'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        
        # Drone position
        p1 = Point()
        p1.x = float(self.state[0])
        p1.y = float(self.state[1])
        p1.z = float(self.state[2])
        
        # Target position
        p2 = Point()
        p2.x = float(self.target_position[0])
        p2.y = float(self.target_position[1])
        p2.z = float(self.target_position[2])
        
        line.points = [p1, p2]
        line.scale.x = 0.02
        
        # Color based on distance (green = close, red = far)
        distance = np.linalg.norm(self.state[:3] - self.target_position)
        line.color = ColorRGBA(
            r=min(1.0, distance / 5.0),
            g=max(0.0, 1.0 - distance / 5.0),
            b=0.0,
            a=0.5
        )
        
        marker_array.markers.append(line)
        
        self.viz_marker_pub.publish(marker_array)
    
    def publish_episode_info(self):
        """Publish episode information as text"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'episode_info'
        marker.id = 3
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 5.0
        marker.pose.orientation.w = 1.0
        
        marker.text = f"Episode: {self.episode_number}\nTarget: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}, {self.target_position[2]:.1f})"
        marker.scale.z = 0.5
        
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.viz_marker_pub.publish(marker_array)
    
    def close(self):
        """Clean up resources"""
        self.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Test with visualization
    env = DroneEnvWithViz(target_position=np.array([5., 0., 2.]), visualize=True)
    obs, info = env.reset()
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, distance={info['distance_to_target']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
