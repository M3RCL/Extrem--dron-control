#!/usr/bin/env python3
"""
Tuned Drone Controller with Hierarchical PID Control (for Simulation Stability)

- Slower, smoother response suitable for RL training in simulation.
- Gains tuned conservatively to avoid instability.
- Uses realistic max tilt and thrust limits.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, FluidPressure, MagneticField
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from actuator_msgs.msg import Actuators
from scipy.spatial.transform import Rotation as R
import time


# ============================================================================
# PID Controller with Improved Anti-Windup
# ============================================================================

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        p = self.kp * error

        # Integral with anti-windup
        self.integral += error * dt
        if self.output_limits and self.ki != 0:
            # Prevent integral from exceeding what output limits would allow
            i_max = self.output_limits[1] / self.ki
            i_min = self.output_limits[0] / self.ki
            self.integral = np.clip(self.integral, i_min, i_max)
        i = self.ki * self.integral

        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        output = p + i + d

        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# ============================================================================
# Velocity-Based Hierarchical Controller (Outer + Inner Loop)
# ============================================================================

class VelocityController:
    def __init__(self):
        self.mass = 1.5
        self.gravity = 9.81
        self.hover_thrust = self.mass * self.gravity  # ~14.7 N
        self.max_thrust = 20.0  # Conservative max thrust (not 25N)
        self.max_tilt = np.radians(15)  # Limit to 15° for stability

        # Outer Loop: Velocity → Desired attitude/thrust
        # Much slower tuning to avoid jerky motion
        self.vel_x_pid = PIDController(
            kp=1.2, ki=0.1, kd=0.3,
            output_limits=(-self.max_tilt, self.max_tilt)
        )
        self.vel_y_pid = PIDController(
            kp=1.2, ki=0.1, kd=0.3,
            output_limits=(-self.max_tilt, self.max_tilt)
        )
        self.vel_z_pid = PIDController(
            kp=2.0, ki=0.2, kd=0.5,
            output_limits=(0.0, self.max_thrust - self.hover_thrust)  # Only regulate delta
        )

        # Inner Loop: Attitude → Torques
        self.roll_pid = PIDController(
            kp=2.0, ki=0.1, kd=0.2,
            output_limits=(-2.0, 2.0)  # N·m
        )
        self.pitch_pid = PIDController(
            kp=2.0, ki=0.1, kd=0.2,
            output_limits=(-2.0, 2.0)
        )
        self.yaw_rate_pid = PIDController(
            kp=0.8, ki=0.05, kd=0.05,
            output_limits=(-1.0, 1.0)
        )

    def reset(self):
        for pid in [self.vel_x_pid, self.vel_y_pid, self.vel_z_pid,
                    self.roll_pid, self.pitch_pid, self.yaw_rate_pid]:
            pid.reset()

    def compute_control(self, desired_vel, current_state, dt):
        if dt <= 0:
            dt = 0.01

        vx, vy, vz, yaw_rate_cmd = desired_vel
        curr_vel = current_state['velocity']
        euler = current_state['euler']
        ang_vel = current_state['angular_velocity']

        # Outer loop: velocity errors → desired attitude/thrust
        des_pitch = -self.vel_x_pid.update(vx - curr_vel[0], dt)
        des_roll = self.vel_y_pid.update(vy - curr_vel[1], dt)
        thrust_delta = self.vel_z_pid.update(vz - curr_vel[2], dt)
        thrust = self.hover_thrust + thrust_delta
        thrust = np.clip(thrust, 0.0, self.max_thrust)

        # Inner loop: attitude errors → torques
        roll_torque = self.roll_pid.update(des_roll - euler[0], dt)
        pitch_torque = self.pitch_pid.update(des_pitch - euler[1], dt)
        yaw_torque = self.yaw_rate_pid.update(yaw_rate_cmd - ang_vel[2], dt)

        return np.array([thrust, roll_torque, pitch_torque, yaw_torque])


class AttitudeController:
    def __init__(self):
        self.mass = 1.5
        self.gravity = 9.81
        self.hover_thrust = self.mass * self.gravity
        self.max_thrust = 20.0

        # Direct attitude control (less aggressive)
        self.roll_pid = PIDController(kp=1.5, ki=0.05, kd=0.1, output_limits=(-1.5, 1.5))
        self.pitch_pid = PIDController(kp=1.5, ki=0.05, kd=0.1, output_limits=(-1.5, 1.5))
        self.yaw_rate_pid = PIDController(kp=0.6, ki=0.02, kd=0.02, output_limits=(-0.8, 0.8))

    def reset(self):
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_rate_pid]:
            pid.reset()

    def compute_control(self, desired_att, current_state, dt):
        if dt <= 0:
            dt = 0.01

        thrust_norm, roll_cmd, pitch_cmd, yaw_rate_cmd = desired_att
        thrust = thrust_norm * self.max_thrust  # Assumes thrust_norm ∈ [0,1]
        roll_cmd = roll_cmd * np.pi /3
        pitch_cmd = pitch_cmd * np.pi /3
        yaw_rate_cmd = yaw_rate_cmd

        euler = current_state['euler']
        ang_vel = current_state['angular_velocity']

        roll_torque = self.roll_pid.update(roll_cmd - euler[0], dt)
        pitch_torque = self.pitch_pid.update(pitch_cmd - euler[1], dt)
        yaw_torque = self.yaw_rate_pid.update(yaw_rate_cmd - ang_vel[2], dt)

        return np.array([thrust, roll_torque, pitch_torque, yaw_torque])


# ============================================================================
# ROS2 Node
# ============================================================================

class DroneControllerWithPID(Node):
    def __init__(self, control_mode='velocity'):
        super().__init__('drone_controller_pid')
        self.declare_parameter('control_mode', control_mode)
        self.control_mode = self.get_parameter('control_mode').value

        self.get_logger().info(f'Control mode: {self.control_mode}')

        # Publishers
        self.motor_pub = self.create_publisher(Actuators, 'x500/command/motor_speed1', 10)
        self.state_pub = self.create_publisher(Float32MultiArray, 'x500/state', 10)

        # Subscribers
        self.create_subscription(Imu, 'x500/imu', self.imu_callback, 10)
        self.create_subscription(FluidPressure, 'x500/air', self.barometer_callback, 10)
        self.create_subscription(MagneticField, 'x500/mag', self.magnetometer_callback, 10)
        self.create_subscription(Odometry, '/model/x500_base/odometry', self.odom_callback, 10)
        self.create_subscription(Float32MultiArray, 'x500/action', self.action_callback, 10)

        # State
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0., 0., 0., 1.])
        self.angular_velocity = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        self.euler_angles = np.zeros(3)
        self.air_pressure = 101325.0
        self.altitude_baro = 0.0
        self.magnetic_field = np.zeros(3)
        self.heading_mag = 0.0

        # Controller
        if self.control_mode == 'velocity':
            self.controller = VelocityController()
        elif self.control_mode == 'attitude':
            self.controller = AttitudeController()
        else:
            raise ValueError(f"Invalid mode: {self.control_mode}")

        # Motor parameters
        self.max_motor_speed = 1500.0
        self.min_motor_speed = 0.0

        # Mixing matrix (X-quad): thrust, roll, pitch, yaw → motors
        # Normalized so that unit inputs produce reasonable outputs
        self.mixing_matrix = np.array([
            [1.0,  1.0,  1.0, -1.0],  # FR (CCW)
            [1.0,  1.0, -1.0,  1.0],  # BL (CCW)
            [1.0, -1.0, -1.0, -1.0],  # FL (CW)
            [1.0, -1.0,  1.0,  1.0]   # BR (CW)
        ]) * 0.25  # Normalize so sum of columns = 1 for thrust

        self.last_time = time.time()
        self.create_timer(0.02, self.publish_state)  # 50 Hz state pub

        self.get_logger().info('Drone Controller (Tuned) Initialized!')

    def imu_callback(self, msg):
        self.orientation = np.array([msg.orientation.x, msg.orientation.y,
                                     msg.orientation.z, msg.orientation.w])
        r = R.from_quat(self.orientation)
        self.euler_angles = r.as_euler('xyz')
        self.angular_velocity = np.array([msg.angular_velocity.x,
                                          msg.angular_velocity.y,
                                          msg.angular_velocity.z])
        self.linear_acceleration = np.array([msg.linear_acceleration.x,
                                             msg.linear_acceleration.y,
                                             msg.linear_acceleration.z])

    def barometer_callback(self, msg):
        self.air_pressure = msg.fluid_pressure
        P0 = 101325.0
        self.altitude_baro = 44330.0 * (1.0 - (self.air_pressure / P0) ** 0.1903)

    def magnetometer_callback(self, msg):
        self.magnetic_field = np.array([msg.magnetic_field.x,
                                        msg.magnetic_field.y,
                                        msg.magnetic_field.z])
        self.heading_mag = np.arctan2(self.magnetic_field[1], self.magnetic_field[0])

    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x,
                                  msg.pose.pose.position.y,
                                  msg.pose.pose.position.z])
        self.velocity = np.array([msg.twist.twist.linear.x,
                                  msg.twist.twist.linear.y,
                                  msg.twist.twist.linear.z])

    def action_callback(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn(f'Expected 4 actions, got {len(msg.data)}')
            return

        action = np.array(msg.data)
        current_time = time.time()
        dt = current_time - self.last_time
        dt = np.clip(dt, 0.001, 0.1)  # avoid dt blowup
        self.last_time = current_time

        current_state = {
            'position': self.position,
            'velocity': self.velocity,
            'euler': self.euler_angles,
            'angular_velocity': self.angular_velocity
        }

        control = self.controller.compute_control(action, current_state, dt)
        motor_speeds = self.control_to_motors(control)
        self.publish_motor_commands(motor_speeds)

    def control_to_motors(self, control):
        thrust, roll_t, pitch_t, yaw_t = control
        
        # Scale inputs to roughly match motor range
        # These scale factors are critical—tune if drone is too sluggish/aggressive
        u1 = thrust / 20.0        # normalize by expected max thrust
        u2 = roll_t / 2.0         # torque scaling
        u3 = pitch_t / 2.0
        u4 = yaw_t / 1.0

        motor_cmds = self.mixing_matrix @ np.array([u1, u2, u3, u4])
        motor_speeds = motor_cmds * self.max_motor_speed
        motor_speeds = np.clip(motor_speeds, self.min_motor_speed, self.max_motor_speed)
        return motor_speeds

    def publish_motor_commands(self, speeds):
        msg = Actuators()
        msg.velocity = speeds.tolist()
        self.motor_pub.publish(msg)

    def publish_state(self):
        state = np.concatenate([
            self.position,
            self.velocity,
            self.euler_angles,
            self.angular_velocity,
            [self.altitude_baro],
            [self.heading_mag]
        ])
        msg = Float32MultiArray(data=state.tolist())
        self.state_pub.publish(msg)

    def reset_controller(self):
        self.controller.reset()
        self.last_time = time.time()


def main(args=None):
    rclpy.init(args=args)
    controller = DroneControllerWithPID(control_mode='velocity')
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()