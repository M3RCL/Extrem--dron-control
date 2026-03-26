"""
real_world_agent.py — Real-hardware inference loop for the trained SAC policy
==============================================================================

No AirSim dependency.  All AirSim calls have been replaced with hardware
abstraction stubs that you implement once for your platform.

Usage:
    python real_world_agent.py \
        --checkpoint checkpoints/latest.pth \
        --goal 5.0 0.0 -1.5

Required hardware:
    - Forward-facing depth camera (planar depth, metric, any resolution)
    - IMU or flight-controller state (orientation quaternion, angular velocity,
      linear velocity, linear acceleration with gravity removed)
    - Flight controller accepting roll/pitch/yaw_rate/throttle commands

Coordinate convention: NED (North=+X, East=+Y, Down=+Z), same as training.
Control frequency: 50 Hz.
"""

import argparse
import logging
import math
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np

# ── Agent (no AirSim imports) ─────────────────────────────────────────────
from sac_multitask import SAC, SACConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")


# =============================================================================
# SECTION 1 — Hardware abstraction layer
# =============================================================================
# Implement the five methods below for your specific platform.
# The rest of this file never calls anything hardware-specific directly.
# =============================================================================

class DroneHardware:
    """
    Abstract interface to real drone hardware.

    Replace each method body with calls to your SDK / ROS subscriber / MAVLink
    library.  Return types and units must match exactly as documented.

    Coordinate frame for all vectors: NED (North=+X, East=+Y, Down=+Z).
    """

    def __init__(self):
        # Put your SDK initialisation here, e.g.:
        #   self.vehicle = dronekit.connect(...)
        #   self.camera  = RealSenseCamera()
        #   self.fc      = MAVLinkController(...)
        pass

    # ------------------------------------------------------------------
    def get_depth_image(self) -> np.ndarray:
        """
        Return the latest planar depth image in METRES as a 2D float32 array.

        Shape: (H, W) — any resolution; the pipeline resizes to (64, 64).
        Values: metric distance in metres (0 = sensor surface, inf/nan = no return).

        Example (Intel RealSense D435):
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale
            return depth_image.astype(np.float32)
        """
        raise NotImplementedError("Implement get_depth_image() for your camera")

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        Return the current drone state as a dict with the following keys.

        All vectors in NED world frame EXCEPT omega_body (body frame).

        Keys:
          pos_world  : np.ndarray (3,) — position in metres from a fixed origin
          vel_world  : np.ndarray (3,) — linear velocity in m/s
          acc_world  : np.ndarray (3,) — linear acceleration in m/s²
                       GRAVITY MUST BE REMOVED (net aerodynamic acceleration only)
          ori_quat   : np.ndarray (4,) — orientation [w, x, y, z]
          omega_body : np.ndarray (3,) — angular velocity in rad/s, body frame

        Example (ArduPilot via dronekit):
            v   = self.vehicle
            att = v.attitude
            # quaternion from Euler — or use your AHRS quaternion directly
            ...
        """
        raise NotImplementedError("Implement get_state() for your flight controller")

    # ------------------------------------------------------------------
    def send_command(self, roll: float, pitch: float,
                     yaw_rate: float, throttle: float) -> None:
        """
        Send a roll/pitch/yaw_rate/throttle command to the flight controller.

        Args:
            roll      : rad, positive = right wing down  (range ± pi/3 = ±60°)
            pitch     : rad, positive = nose down        (range ± pi/3 = ±60°)
            yaw_rate  : rad/s, positive = clockwise from above (range ± pi)
            throttle  : normalised [0, 5.0] — map to your FC's throttle scale

        The agent runs at 50 Hz.  This method is called once per control cycle.
        It should be non-blocking (fire-and-forget or async).

        Example (MAVLink / pymavlink):
            self.master.mav.set_attitude_target_send(
                0,                          # time_boot_ms
                self.master.target_system,
                self.master.target_component,
                0b00000000,                 # type_mask: use everything
                quaternion_from_euler(roll, pitch, 0),
                yaw_rate,
                0, 0,                       # roll_rate, pitch_rate
                throttle / 5.0,             # normalise to [0, 1]
            )
        """
        raise NotImplementedError("Implement send_command() for your flight controller")

    # ------------------------------------------------------------------
    def send_stop(self) -> None:
        """
        Command the drone to hover / hold position immediately.
        Called on Ctrl-C, error, or safety timeout.

        Example:
            self.vehicle.mode = dronekit.VehicleMode("LOITER")
        """
        raise NotImplementedError("Implement send_stop() for your flight controller")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """
        Release all hardware resources (close connections, stop threads, etc.).
        """
        pass


# =============================================================================
# SECTION 2 — Observation pipeline
# =============================================================================
# Identical to the training pipeline in drone_obstacle_env.py.
# Do NOT change anything here — any deviation breaks the policy.
# =============================================================================

# Training constants — must not be changed
_MAX_DEPTH   = 50.0    # metres — normalisation denominator
_FRAME_STACK = 4       # depth frames stacked
_VEC_STACK   = 3       # proprioceptive history frames
_VEC_DIM     = 12      # dims per proprioceptive frame
_ACTION_DIM  = 4


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Body rotation matrix R such that v_body = R.T @ v_world."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _process_depth(raw: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a raw depth image to match training.

    raw  : (H, W) float32, metric metres
    out  : (64, 64) float32, values in [0, 1]
           1.0 = obstacle at contact distance
           0.0 = free space at >= 50 m

    NaN and inf in the raw image are treated as maximum range (0.0 output).
    """
    raw = np.nan_to_num(raw, nan=_MAX_DEPTH, posinf=_MAX_DEPTH)
    resized = cv2.resize(raw, (64, 64), interpolation=cv2.INTER_AREA)
    return (1.0 - np.clip(resized / _MAX_DEPTH, 0.0, 1.0)).astype(np.float32)


def _build_vec_frame(goal_world: np.ndarray, state: dict) -> np.ndarray:
    """
    Compute one 12-dim proprioceptive frame from the current drone state.

    Layout  : [goal_body(3) | vel_body(3) | omega_body(3) | acc_body(3)]
    """
    R           = _rotation_matrix(state["ori_quat"])
    goal_body   = (R.T @ (goal_world - state["pos_world"])).astype(np.float32)
    vel_body    = (R.T @ state["vel_world"]).astype(np.float32)
    omega_body  = state["omega_body"].astype(np.float32)   # already body frame
    acc_body    = (R.T @ state["acc_world"]).astype(np.float32)  # gravity removed

    return np.concatenate([goal_body, vel_body, omega_body, acc_body])


class ObservationBuilder:
    """
    Maintains the rolling depth buffer and proprioceptive history deque
    and assembles the obs dict expected by SAC.select_action().
    """

    def __init__(self):
        blank_depth = np.zeros((64, 64), dtype=np.float32)
        blank_vec   = np.zeros(_VEC_DIM, dtype=np.float32)
        self._depth_buf = deque(
            [blank_depth.copy() for _ in range(_FRAME_STACK)], maxlen=_FRAME_STACK
        )
        self._vec_buf = deque(
            [blank_vec.copy() for _ in range(_VEC_STACK)], maxlen=_VEC_STACK
        )
        self._prev_action = np.zeros(_ACTION_DIM, dtype=np.float32)

    def reset(self):
        """Call once before each flight / mission start."""
        blank_depth = np.zeros((64, 64), dtype=np.float32)
        blank_vec   = np.zeros(_VEC_DIM, dtype=np.float32)
        self._depth_buf = deque(
            [blank_depth.copy() for _ in range(_FRAME_STACK)], maxlen=_FRAME_STACK
        )
        self._vec_buf = deque(
            [blank_vec.copy() for _ in range(_VEC_STACK)], maxlen=_VEC_STACK
        )
        self._prev_action = np.zeros(_ACTION_DIM, dtype=np.float32)

    def update(self, raw_depth: np.ndarray, goal_world: np.ndarray,
               state: dict) -> dict:
        """
        Process new sensor data and return the obs dict.

        Must be called exactly once per control cycle (50 Hz).
        """
        # Depth
        self._depth_buf.append(_process_depth(raw_depth))

        # Proprioceptive frame
        self._vec_buf.append(_build_vec_frame(goal_world, state))

        depth_stack = np.array(list(self._depth_buf), dtype=np.float32)  # (4,64,64)
        vec_hist    = np.concatenate(list(self._vec_buf)).astype(np.float32)  # (36,)

        return {
            "depth":       depth_stack,
            "vec_hist":    vec_hist,
            "prev_action": self._prev_action.copy(),
        }

    def record_action(self, action: np.ndarray):
        """Call after every select_action() to update prev_action for next step."""
        self._prev_action = action.astype(np.float32)


# =============================================================================
# SECTION 3 — Action scaling
# =============================================================================
# Converts the agent's [-1, 1] output to physical flight-controller commands.
# These constants match EnvConfig in drone_obstacle_env.py exactly.
# =============================================================================

_MAX_ROLL     = math.pi / 3   # 60°
_MAX_PITCH    = math.pi / 3   # 60°
_MAX_YAW_RATE = math.pi       # 180°/s
_MAX_THROTTLE = 5.0           # normalised throttle ceiling


def scale_action(action: np.ndarray):
    """
    Scale agent output → physical commands.

    action : (4,) in [-1, 1]   [roll_n, pitch_n, yaw_rate_n, throttle_n]

    Returns:
        roll      : rad  ∈ [−π/3, +π/3]
        pitch     : rad  ∈ [−π/3, +π/3]
        yaw_rate  : rad/s ∈ [−π, +π]
        throttle  : float ∈ [0, 5.0]
    """
    roll      = float(action[0]) * _MAX_ROLL
    pitch     = float(action[1]) * _MAX_PITCH
    yaw_rate  = float(action[2]) * _MAX_YAW_RATE
    throttle  = (float(action[3]) + 1.0) / 2.0 * _MAX_THROTTLE
    return roll, pitch, yaw_rate, throttle


# =============================================================================
# SECTION 4 — Safety watchdog
# =============================================================================

class SafetyWatchdog:
    """
    Lightweight safety layer.  Checks simple conditions every step and triggers
    a stop if any are violated.  Extend check() for your own constraints.
    """

    def __init__(self, arena_bounds: dict, max_speed_mps: float = 15.0):
        """
        arena_bounds : dict with keys x_min, x_max, y_min, y_max, z_min, z_max (NED metres)
        max_speed_mps: cut-out if speed exceeds this threshold
        """
        self.bounds       = arena_bounds
        self.max_speed    = max_speed_mps
        self.triggered    = False
        self.trigger_reason = ""

    def check(self, state: dict) -> bool:
        """
        Returns True if everything is safe, False if the watchdog has triggered.
        Once triggered it stays triggered — call reset() to clear.
        """
        if self.triggered:
            return False

        pos = state["pos_world"]
        vel = state["vel_world"]
        b   = self.bounds

        if not (b["x_min"] <= pos[0] <= b["x_max"] and
                b["y_min"] <= pos[1] <= b["y_max"] and
                b["z_min"] <= pos[2] <= b["z_max"]):
            self._trigger(f"Out of bounds: pos={pos}")
            return False

        speed = float(np.linalg.norm(vel))
        if speed > self.max_speed:
            self._trigger(f"Speed too high: {speed:.1f} m/s")
            return False

        return True

    def _trigger(self, reason: str):
        self.triggered = True
        self.trigger_reason = reason
        log.error(f"SAFETY WATCHDOG: {reason}")

    def reset(self):
        self.triggered = False
        self.trigger_reason = ""


# =============================================================================
# SECTION 5 — Main flight loop
# =============================================================================

class RealWorldRunner:
    """
    Ties together hardware, observation pipeline, agent, and safety watchdog
    into a 50 Hz closed-loop controller.
    """

    DT = 0.02   # 50 Hz — must match training

    def __init__(self, hardware: DroneHardware, agent: SAC,
                 goal_world: np.ndarray, watchdog: SafetyWatchdog,
                 deterministic: bool = True, max_steps: int = 1_000):
        self.hw          = hardware
        self.agent       = agent
        self.goal        = goal_world.astype(np.float64)
        self.watchdog    = watchdog
        self.obs_builder = ObservationBuilder()
        self.deterministic = deterministic
        self.max_steps   = max_steps

        self._running    = False
        self._step       = 0

        # Metrics collected during the run
        self.metrics = {
            "steps": 0,
            "dist_to_goal_final": None,
            "min_dist_to_goal": float("inf"),
            "termination": "not_started",
            "action_log": [],   # list of (step, roll, pitch, yaw_rate, throttle)
        }

        # Register Ctrl-C handler
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        log.warning("Interrupt received — stopping drone")
        self._running = False

    def _reach_check(self, pos_world: np.ndarray, threshold_m: float = 0.5) -> bool:
        dist = float(np.linalg.norm(self.goal - pos_world))
        if dist < self.metrics["min_dist_to_goal"]:
            self.metrics["min_dist_to_goal"] = dist
        return dist < threshold_m

    def prime_buffers(self, n_frames: int = 4):
        """
        Collect n_frames observations before releasing control to the agent.
        This warms up the depth buffer with real sensor data instead of zeros.
        Call this while the drone is stationary at the launch position.
        """
        log.info(f"Priming observation buffers ({n_frames} frames) …")
        for i in range(n_frames):
            raw_depth = self.hw.get_depth_image()
            state     = self.hw.get_state()
            self.obs_builder.update(raw_depth, self.goal, state)
            time.sleep(self.DT)
        log.info("Buffers primed.")

    def run(self) -> dict:
        """
        Execute the closed-loop control at 50 Hz until success, safety stop,
        or max_steps is reached.

        Returns the metrics dict.
        """
        self.obs_builder.reset()
        self.prime_buffers()
        self._running = True
        self._step    = 0
        self.metrics["termination"] = "timeout"

        log.info(f"Starting control loop | goal={self.goal} | "
                 f"deterministic={self.deterministic} | max_steps={self.max_steps}")

        while self._running and self._step < self.max_steps:
            t_start = time.perf_counter()
            self._step += 1

            # ── 1. Sense ─────────────────────────────────────────────────
            raw_depth = self.hw.get_depth_image()
            state     = self.hw.get_state()

            # ── 2. Safety check ──────────────────────────────────────────
            if not self.watchdog.check(state):
                self.metrics["termination"] = "safety_stop"
                self._running = False
                break

            # ── 3. Build observation ─────────────────────────────────────
            obs = self.obs_builder.update(raw_depth, self.goal, state)

            # ── 4. Inference ─────────────────────────────────────────────
            action = self.agent.select_action(obs, deterministic=self.deterministic)
            self.obs_builder.record_action(action)

            # ── 5. Scale and send ─────────────────────────────────────────
            roll, pitch, yaw_rate, throttle = scale_action(action)
            self.hw.send_command(roll, pitch, yaw_rate, throttle)

            # ── 6. Logging ────────────────────────────────────────────────
            dist = float(np.linalg.norm(self.goal - state["pos_world"]))
            if dist < self.metrics["min_dist_to_goal"]:
                self.metrics["min_dist_to_goal"] = dist
            self.metrics["action_log"].append(
                (self._step, roll, pitch, yaw_rate, throttle))

            if self._step % 50 == 0:
                log.info(
                    f"step={self._step:4d}  dist={dist:6.2f}m  "
                    f"pos=({state['pos_world'][0]:5.1f}, "
                    f"{state['pos_world'][1]:5.1f}, "
                    f"{state['pos_world'][2]:5.1f})  "
                    f"R={math.degrees(roll):5.1f}°  "
                    f"P={math.degrees(pitch):5.1f}°  "
                    f"thr={throttle:.2f}"
                )

            # ── 7. Goal check ─────────────────────────────────────────────
            if self._reach_check(state["pos_world"], threshold_m=0.5):
                log.info(f"Goal reached at step {self._step}  dist={dist:.3f} m")
                self.metrics["termination"] = "success"
                self._running = False
                break

            # ── 8. Pace the loop to exactly 50 Hz ────────────────────────
            elapsed = time.perf_counter() - t_start
            remaining = self.DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
            elif elapsed > self.DT * 1.5:
                log.warning(
                    f"Step {self._step}: loop overran by "
                    f"{(elapsed - self.DT)*1000:.1f} ms"
                )

        # ── Shutdown ──────────────────────────────────────────────────────
        log.info(f"Sending stop command (termination={self.metrics['termination']})")
        self.hw.send_stop()

        state = self.hw.get_state()
        self.metrics["steps"]             = self._step
        self.metrics["dist_to_goal_final"] = float(
            np.linalg.norm(self.goal - state["pos_world"]))

        log.info(
            f"Run complete | termination={self.metrics['termination']} | "
            f"steps={self._step} | "
            f"dist_final={self.metrics['dist_to_goal_final']:.2f} m | "
            f"dist_min={self.metrics['min_dist_to_goal']:.2f} m"
        )
        return self.metrics


# =============================================================================
# SECTION 6 — Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-hardware inference for the trained drone SAC policy"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained checkpoint (e.g. checkpoints/latest.pth)")
    parser.add_argument("--goal", nargs=3, type=float, required=True,
                        metavar=("X", "Y", "Z"),
                        help="Goal position in NED metres (e.g. --goal 10.0 0.0 -1.5)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum control steps before timeout (default 1000 = 20 s)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--x-min", type=float, default=-30.0)
    parser.add_argument("--x-max", type=float, default= 30.0)
    parser.add_argument("--y-min", type=float, default=-30.0)
    parser.add_argument("--y-max", type=float, default= 30.0)
    parser.add_argument("--z-min", type=float, default=-25.0)
    parser.add_argument("--z-max", type=float, default= -2.0)
    args = parser.parse_args()

    goal_world = np.array(args.goal, dtype=np.float64)
    log.info(f"Goal (NED): {goal_world}")

    # ── Load agent ────────────────────────────────────────────────────────
    log.info("Loading SAC agent …")
    agent = SAC(
        frame_stack  = 4,
        image_size   = 64,
        action_dim   = 4,
        vec_hist_dim = 36,   # vec_stack(3) × 12
        config       = SACConfig(),
    )
    agent.load(args.checkpoint)
    log.info("Agent loaded.")

    # ── Initialise hardware ───────────────────────────────────────────────
    log.info("Initialising hardware …")
    hw = DroneHardware()

    # ── Safety watchdog ───────────────────────────────────────────────────
    watchdog = SafetyWatchdog(
        arena_bounds={
            "x_min": args.x_min, "x_max": args.x_max,
            "y_min": args.y_min, "y_max": args.y_max,
            "z_min": args.z_min, "z_max": args.z_max,
        }
    )

    # ── Run ───────────────────────────────────────────────────────────────
    runner = RealWorldRunner(
        hardware      = hw,
        agent         = agent,
        goal_world    = goal_world,
        watchdog      = watchdog,
        deterministic = not args.stochastic,
        max_steps     = args.max_steps,
    )

    try:
        metrics = runner.run()
    except Exception as exc:
        log.exception(f"Unexpected error during flight: {exc}")
        hw.send_stop()
        metrics = runner.metrics
    finally:
        hw.close()

    print("\n=== Flight summary ===")
    print(f"  Termination     : {metrics['termination']}")
    print(f"  Steps           : {metrics['steps']}")
    print(f"  Dist final      : {metrics['dist_to_goal_final']:.2f} m")
    print(f"  Dist min        : {metrics['min_dist_to_goal']:.2f} m")


if __name__ == "__main__":
    main()