#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    pkg_x500_sim = FindPackageShare('x500_simulation')
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')
    
    world_file = PathJoinSubstitution([pkg_x500_sim, 'worlds', 'drone_world.sdf'])
    bridge_config = PathJoinSubstitution([pkg_x500_sim, 'config', 'bridge.yaml'])
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_file],
            'on_exit_shutdown': 'true'
        }.items()
    )
    
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p', f'config_file:={bridge_config}',
        ],
        output='screen'
    )
    
    controller = Node(
        package='x500_simulation',
        executable='drone_controller.py',
        name='drone_controller',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    tf_world_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        gazebo_sim,
        bridge,
        controller,
        tf_world_to_base,
    ])
