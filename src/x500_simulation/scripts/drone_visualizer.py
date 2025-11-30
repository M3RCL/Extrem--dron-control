#!/usr/bin/env python3
"""
Drone Visualization Node for RViz
Publishes markers, paths, and transforms for visualization during training
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32MultiArray
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


class DroneVisualizer(Node):
    def __init__(self):
        super().__init__('drone_visualizer')
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_markers', 10)
        self.path_pub = self.create_publisher(Path, 'drone_path', 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/model/x500/odometry', self.odom_callback, 10
        )
        self.state_sub = self.create_subscription(
            Float32MultiArray, 'x500/state', self.state_callback, 10
        )
        
        # State storage
        self.current_pose = None
        self.current_state = None
        self.path_points = []
        self.max_path_points = 500
        
        # Target position (set this from your RL environment)
        self.target_position = np.array([5.0, 0.0, 2.0])
        
        # Visualization timer
        self.vis_timer = self.create_timer(0.1, self.publish_visualizations)
        
        self.get_logger().info('Drone Visualizer initialized')
    
    def odom_callback(self, msg):
        """Store odometry data for visualization"""
        self.current_pose = msg.pose.pose
        
        # Add to path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'world'
        pose_stamped.pose = self.current_pose
        
        self.path_points.append(pose_stamped)
        if len(self.path_points) > self.max_path_points:
            self.path_points.pop(0)
        
        # Publish TF
        self.publish_tf(msg)
    
    def state_callback(self, msg):
        """Store state data"""
        if len(msg.data) >= 12:
            self.current_state = np.array(msg.data[:12])
    
    def publish_tf(self, odom_msg):
        """Publish transform from world to base_link"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        
        t.transform.rotation = odom_msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_visualizations(self):
        """Publish all visualization markers"""
        if self.current_pose is None:
            return
        
        marker_array = MarkerArray()
        
        # Marker 1: Target position sphere
        marker_array.markers.append(self.create_target_marker())
        
        # Marker 2: Velocity vector
        if self.current_state is not None:
            marker_array.markers.append(self.create_velocity_arrow())
        
        # Marker 3: Distance text
        marker_array.markers.append(self.create_distance_text())
        
        # Marker 4: Orientation indicator
        marker_array.markers.append(self.create_orientation_cube())
        
        # Marker 5: Sensor range sphere
        marker_array.markers.append(self.create_sensor_range())
        
        # Publish markers
        self.marker_pub.publish(marker_array)
        
        # Publish path
        self.publish_path()
    
    def create_target_marker(self):
        """Create target position marker"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.target_position[0])
        marker.pose.position.y = float(self.target_position[1])
        marker.pose.position.z = float(self.target_position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7)
        marker.lifetime.sec = 0  # Persistent
        
        return marker
    
    def create_velocity_arrow(self):
        """Create velocity vector arrow"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'velocity'
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point (drone position)
        start = Point()
        start.x = self.current_pose.position.x
        start.y = self.current_pose.position.y
        start.z = self.current_pose.position.z
        
        # End point (drone position + velocity)
        vel = self.current_state[3:6]
        end = Point()
        end.x = start.x + vel[0]
        end.y = start.y + vel[1]
        end.z = start.z + vel[2]
        
        marker.points = [start, end]
        
        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1   # Head diameter
        marker.scale.z = 0.0
        
        # Color based on velocity magnitude
        vel_mag = np.linalg.norm(vel)
        marker.color = ColorRGBA(
            r=min(1.0, vel_mag / 5.0),
            g=0.0,
            b=max(0.0, 1.0 - vel_mag / 5.0),
            a=0.8
        )
        
        return marker
    
    def create_distance_text(self):
        """Create text showing distance to target"""
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'distance_text'
        marker.id = 2
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Calculate distance
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z
        ])
        distance = np.linalg.norm(current_pos - self.target_position)
        
        marker.pose.position.x = self.current_pose.position.x
        marker.pose.position.y = self.current_pose.position.y
        marker.pose.position.z = self.current_pose.position.z + 0.5
        marker.pose.orientation.w = 1.0
        
        marker.text = f"Distance: {distance:.2f}m"
        marker.scale.z = 0.3  # Text height
        
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        
        return marker
    
    def create_orientation_cube(self):
        """Create cube showing drone orientation"""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'orientation'
        marker.id = 3
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.47  # Drone body size
        marker.scale.y = 0.47
        marker.scale.z = 0.11
        
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
        
        return marker
    
    def create_sensor_range(self):
        """Create sphere showing sensor range"""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sensor_range'
        marker.id = 4
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 2.0  # Sensor range
        marker.scale.y = 2.0
        marker.scale.z = 2.0
        
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.1)
        
        return marker
    
    def publish_path(self):
        """Publish drone trajectory path"""
        if len(self.path_points) == 0:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'
        path_msg.poses = self.path_points
        
        self.path_pub.publish(path_msg)
    
    def set_target_position(self, position):
        """Update target position for visualization"""
        self.target_position = np.array(position)


def main(args=None):
    rclpy.init(args=args)
    visualizer = DroneVisualizer()
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
