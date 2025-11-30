#!/usr/bin/env python3
"""
Launch file for X500 drone simulation with RViz visualization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    
    # Package directories
    pkg_x500_sim = FindPackageShare('x500_simulation')
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')
    
    # Paths
    world_file = PathJoinSubstitution([pkg_x500_sim, 'worlds', 'drone_world.sdf'])
    bridge_config = PathJoinSubstitution([pkg_x500_sim, 'config', 'bridge.yaml'])
    rviz_config = PathJoinSubstitution([pkg_x500_sim, 'config', 'drone_rviz.rviz'])
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    
    # Gazebo simulation
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_file],
            'on_exit_shutdown': 'true'
        }.items()
    )
    
    # ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge',
        parameters=[{'config_file':bridge_config}],
        output='screen'
    )
    
    # Drone controller node
    controller = Node(
        package='x500_simulation',
        executable='drone_controller.py',
        name='drone_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    bridge_world_services = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/drone_world/set_pose@ros_gz_interfaces/srv/SetEntityPose',
            '/world/drone_world/entity_system@ros_gz_interfaces/srv/EntityInfo',
        ],
        output='screen'
    )
    # Drone visualizer node
    visualizer = Node(
        package='x500_simulation',
        executable='drone_visualizer.py',
        name='drone_visualizer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz)
    )
    
    # RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen'
    )
    
    # TF2 static transform (world to odom, if needed)
    tf_world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation time'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Launch RViz visualization'),
        gazebo_sim,
        bridge,
        bridge_world_services,
        controller,
        visualizer,
        rviz,
        tf_world_to_odom,
    ])
