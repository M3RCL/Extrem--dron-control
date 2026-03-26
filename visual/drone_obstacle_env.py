"""
Drone Obstacle Avoidance Environment
=====================================
Sensors:
  - Depth camera (64x64, stacked x4)
  - IMU  (orientation + angular velocity — body-frame transforms only)
  - Positioning system (GT position + velocity from AirSim kinematics)

Observation space:
  depth      : (frame_stack, H, W)  stacked depth images
  goal_body  : (3,)                 goal vector in body frame
  vel_body   : (3,)                 velocity in body frame
  omega_body : (3,)                 angular velocity in body frame (from IMU)
  prev_action: (4,)

Reward:
  r = w_approach * dot(vel, goal_dir)        [approach velocity]
    + w_accel   * dot(acc, goal_dir)         [radial acceleration]
    + potential_field_obstacles              [repulsive field, negative]
    + time_penalty
  terminal: success +500, collision -500, OOB -250

Curriculum: obstacle count and goal threshold co-vary as a single integer level.
"""

try:
    import cosysairsim as airsim
    from airsim import utils as airsim_utils
    COSYS_AIRSIM = True
    print("Using Cosys-AirSim")
except ImportError:
    import airsim
    airsim_utils = None
    COSYS_AIRSIM = False
    print("Using standard AirSim")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def to_quaternion(pitch: float, roll: float, yaw: float):
    if COSYS_AIRSIM and airsim_utils is not None:
        return airsim_utils.euler_to_quaternion(pitch, roll, yaw)
    return airsim.to_quaternion(pitch, roll, yaw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    # Control
    dt: float = 0.02          # 50 Hz
    max_steps: int = 1000

    # Image
    image_width: int = 64
    image_height: int = 64
    frame_stack: int = 4
    max_depth: float = 50.0

    # Proprioceptive history stacking
    vec_stack: int = 3

    # Action limits
    max_roll: float = math.pi / 3
    max_pitch: float = math.pi / 3
    max_yaw_rate: float = math.pi
    max_throttle: float = 5.0

    # Arena bounds (NED)
    arena_x_min: float = -30.0
    arena_x_max: float = 30.0
    arena_y_min: float = -30.0
    arena_y_max: float = 30.0
    arena_z_min: float = -25.0
    arena_z_max: float = -2.0

    # Obstacles
    # max_obstacles covers the largest ring set: 3 rings × 5 obs = 15, +2 spare slots = 17.
    # _pre_spawn_obstacles uses (max_obstacles + 2) so keep this at 15.
    max_obstacles: int = 15
    obstacle_min_size: float = 1.0
    obstacle_max_size: float = 1.5
    obstacle_height_multiplier: float = 8.0

    # Multi-ring placement around goal.
    # Each entry is the radius (m) of one concentric ring.
    # Curriculum adds rings one at a time starting from the innermost.
    # 5 obstacles are placed on each active ring.
    obstacle_ring_radii: Tuple[float, ...] = (7.0, 14.0, 21.0)
    obstacle_gap_deg: float = 75.0          # angular gap left open per ring (passable corridor)
    obstacle_min_angle_sep_deg: float = 30.0  # min angular separation between obstacles on same ring

    # Drone spawn — relative to the outermost active ring.
    # Drone is placed (outer_radius + offset) metres from the goal along the -X axis,
    # with a small random Y jitter.  This guarantees the drone starts outside all rings
    # and is much closer to the obstacle cluster than the old fixed-zone spawn.
    spawn_offset_min: float = 5.0    # metres beyond outer ring edge
    spawn_offset_max: float = 10.0   # metres beyond outer ring edge

    # Goal marker
    marker_size: float = 1.5

    # Terminal rewards
    collision_penalty: float = -1000.0
    success_reward: float = 500.0
    oob_penalty: float = -500.0

    # Dense reward weights
    w_approach: float = 1.0
    w_accel: float = 1.0
    w_heading: float = 5.0
    w_depth_penalty: float = 4.0
    depth_danger_thresh: float = 0.85
    depth_penalty_sharpness: float = 6.0
    time_penalty: float = -5

    # Braking envelope
    d_brake: float = 2.0

    # Smooth distance bowl
    w_dist: float = 1.0
    k_dist: float = 0.02


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
# Each entry: (num_obstacles, reach_threshold_m)
# Levels 0-6 are the original training curriculum.
# Levels 7-8 are fine-tuning: threshold stays at 0.5 m, rings are added.
#   Level 7 (10 obs) → 2 rings of 5 each (inner r=7 m, middle r=14 m)
#   Level 8 (15 obs) → 3 rings of 5 each (inner + middle + outer r=21 m)
CURRICULUM_LEVELS: List[Tuple[int, float]] = [
    (7,  5.0),   # 0 — no obstacles
    (7,  4.0),   # 1
    (7,  3.0),   # 2
    (7,  2.0),   # 3
    (7,  1.5),   # 4
    (7,  1.0),   # 5
    (7,  0.5),   # 6 — trained state: 1 ring, 5 obstacles, 0.5 m threshold
    (10, 0.5),   # 7 — fine-tune: 2 rings, 10 obstacles
    (15, 0.5),   # 8 — fine-tune: 3 rings, 15 obstacles
]

# Obstacles per ring (constant — each ring always holds this many)
_OBS_PER_RING: int = 5


# ---------------------------------------------------------------------------
# Obstacle Manager  (no corridor walls)
# ---------------------------------------------------------------------------

class ObstacleManager:
    OBSTACLE_ASSET = "EditorCube"  # single asset — reliable scale/pose support
    OBSTACLE_PREFIX = "RLObs_"
    SCENE_OBSTACLE_PATTERNS = ['Cylinder', 'Cone_', 'Cube', 'Sphere']
    PROTECTED_OBJECTS = ['SimpleFlight', 'Ground', 'Sky', 'Light', 'Camera', 'Player',
                         'Menu', 'Fog', 'PostProcess', 'Volume', 'NavData', 'GameMode',
                         'BP_', 'Brush', 'Chaos', 'Game', 'Sim', 'Weather', 'External']
    GOAL_MARKER_NAME = "OrangeBall"
    HIDDEN_POS = airsim.Vector3r(0, 0, 500)

    MIN_OBS_FROM_START: float = 5.0   # min clearance from drone start (m)

    def __init__(self, client, config: EnvConfig):
        self.client = client
        self.config = config
        self.spawned_obstacles: List[str] = []
        self.hidden_scene_objects: List[str] = []
        self.obstacle_positions: List[np.ndarray] = []
        self.obstacle_radii: List[float] = []
        self.goal_marker_exists = False

        print("=" * 60)
        print("OBSTACLE MANAGER INIT")
        print("=" * 60)
        all_objs = self._list_scene_objects()
        self._destroy_leftover(all_objs)
        self._hide_scene_obstacles(all_objs)
        self._check_goal_marker()
        self._pre_spawn_obstacles()
        print("=" * 60)

    def _list_scene_objects(self):
        try:
            return self.client.simListSceneObjects()
        except Exception as e:
            print(f"  simListSceneObjects: {e}")
            return []

    def _is_protected(self, name: str) -> bool:
        return any(p.lower() in name.lower() for p in self.PROTECTED_OBJECTS)

    def _destroy_leftover(self, all_objs):
        leftovers = [o for o in all_objs if o.startswith(self.OBSTACLE_PREFIX)]
        for name in leftovers:
            try:
                self.client.simDestroyObject(name)
            except Exception:
                pass
        print(f"  Removed {len(leftovers)} leftover objects")

    def _hide_scene_obstacles(self, all_objs):
        hidden_pose = airsim.Pose(self.HIDDEN_POS, airsim.Quaternionr(0, 0, 0, 1))
        to_hide = [
            obj for obj in all_objs
            if not self._is_protected(obj)
            and not obj.startswith(self.OBSTACLE_PREFIX)
            and obj != self.GOAL_MARKER_NAME
            and any(obj.startswith(p) for p in self.SCENE_OBSTACLE_PATTERNS)
        ]
        for name in to_hide:
            try:
                self.client.simSetObjectPose(name, hidden_pose, teleport=True)
                self.hidden_scene_objects.append(name)
            except Exception:
                pass
        print(f"  Hid {len(self.hidden_scene_objects)} scene objects")

    def _check_goal_marker(self):
        try:
            pose = self.client.simGetObjectPose(self.GOAL_MARKER_NAME)
            self.goal_marker_exists = (pose.position.x_val == pose.position.x_val)  # NaN check
        except Exception:
            self.goal_marker_exists = False
        print(f"  Goal marker: {'found' if self.goal_marker_exists else 'not found'}")

    def _pre_spawn_obstacles(self):
        # +2 spare slots above the maximum needed (3 rings × 5 = 15)
        pool_size = self.config.max_obstacles + 2
        hidden_pose = airsim.Pose(self.HIDDEN_POS, airsim.Quaternionr(0, 0, 0, 1))
        for i in range(pool_size):
            requested_name = f"{self.OBSTACLE_PREFIX}{i}"
            try:
                returned_name = self.client.simSpawnObject(
                    object_name=requested_name,
                    asset_name=self.OBSTACLE_ASSET,
                    pose=hidden_pose,
                    scale=airsim.Vector3r(1, 1, 1),
                    physics_enabled=False,
                    is_blueprint=False,
                )
                actual_name = (returned_name
                               if (returned_name and isinstance(returned_name, str))
                               else requested_name)
                self.spawned_obstacles.append(actual_name)
                print(f"  Spawned slot {i}: '{actual_name}'")
            except Exception as e:
                print(f"  Spawn error slot {i}: {e}")
        print(f"  Pre-spawned {len(self.spawned_obstacles)}/{pool_size} obstacle slots")

    # ------------------------------------------------------------------
    def _sample_ring_angles(self, n: int, gap_deg: float,
                             min_sep_deg: float) -> list:
        """
        Sample n angles (radians) on a circle with a guaranteed gap of
        `gap_deg` degrees at a random orientation each call.
        Returns up to n angles (may be fewer if the arc is too crowded).
        """
        gap_rad     = math.radians(gap_deg)
        min_sep_rad = math.radians(min_sep_deg)
        gap_center  = np.random.uniform(0, 2 * math.pi)

        arc_start = gap_center + gap_rad / 2.0
        arc_len   = 2 * math.pi - gap_rad

        placed_angles = []
        MAX_TRIES = 200
        tries = 0
        while len(placed_angles) < n and tries < MAX_TRIES:
            tries += 1
            a = (arc_start + np.random.uniform(0, arc_len)) % (2 * math.pi)
            ok = True
            for prev in placed_angles:
                diff = abs(a - prev)
                diff = min(diff, 2 * math.pi - diff)
                if diff < min_sep_rad:
                    ok = False
                    break
            if ok:
                placed_angles.append(a)

        return placed_angles

    # ------------------------------------------------------------------
    def setup_episode(self, num_obstacles: int, drone_start: np.ndarray,
                      goal_pos: np.ndarray) -> None:
        """
        Place obstacles on one, two, or three concentric rings centred on
        the goal, depending on num_obstacles:
          1–5  → 1 ring  (r = obstacle_ring_radii[0])
          6–10 → 2 rings (r = obstacle_ring_radii[0:2])
          11–15→ 3 rings (r = obstacle_ring_radii[0:3])

        Each ring gets an independently randomised gap direction so the
        drone cannot memorise the exit corridor.  The existing
        _sample_ring_angles logic is reused unchanged for each ring.
        """
        self.obstacle_positions = []
        self.obstacle_radii = []
        cfg = self.config

        # Hide all pre-spawned slots first
        hidden_pose = airsim.Pose(self.HIDDEN_POS, airsim.Quaternionr(0, 0, 0, 1))
        for name in self.spawned_obstacles:
            try:
                self.client.simSetObjectPose(name, hidden_pose, teleport=True)
            except Exception:
                pass

        if num_obstacles == 0:
            print("  [ObstacleMgr] 0 obstacles — arena clear")
            return

        # Determine how many rings to activate
        num_rings = math.ceil(num_obstacles / _OBS_PER_RING)
        num_rings = min(num_rings, len(cfg.obstacle_ring_radii))
        active_radii = cfg.obstacle_ring_radii[:num_rings]

        z_center = (cfg.arena_z_min + cfg.arena_z_max) / 2.0
        slot_idx = 0   # index into self.spawned_obstacles
        total_placed = 0

        for ring_idx, ring_radius in enumerate(active_radii):
            # How many obstacles on this ring?  Last ring gets the remainder.
            remaining_obs = num_obstacles - ring_idx * _OBS_PER_RING
            obs_this_ring = min(_OBS_PER_RING, remaining_obs)

            angles = self._sample_ring_angles(
                obs_this_ring, cfg.obstacle_gap_deg, cfg.obstacle_min_angle_sep_deg
            )
            if len(angles) < obs_this_ring:
                print(f"  [ring {ring_idx}] only {len(angles)}/{obs_this_ring} angles fit")

            placed_this_ring = 0

            for angle in angles:
                if slot_idx >= len(self.spawned_obstacles):
                    print(f"  [ring {ring_idx}] ran out of pre-spawned slots")
                    break

                cx = goal_pos[0] + ring_radius * math.cos(angle)
                cy = goal_pos[1] + ring_radius * math.sin(angle)

                # Small jitter so obstacles are not perfectly equidistant
                jitter = np.random.uniform(-1.5, 1.5, size=2)
                cx = float(np.clip(cx + jitter[0], cfg.arena_x_min + 3, cfg.arena_x_max - 3))
                cy = float(np.clip(cy + jitter[1], cfg.arena_y_min + 3, cfg.arena_y_max - 3))
                pos = np.array([cx, cy, z_center])

                # Skip if dangerously close to drone start
                if np.linalg.norm(pos[:2] - drone_start[:2]) < self.MIN_OBS_FROM_START:
                    print(f"  [ring {ring_idx}] angle {math.degrees(angle):.0f}° "
                          f"too close to drone start — skipping")
                    slot_idx += 1
                    continue

                size   = np.random.uniform(cfg.obstacle_min_size, cfg.obstacle_max_size)
                height = size * cfg.obstacle_height_multiplier
                name   = self.spawned_obstacles[slot_idx]
                slot_idx += 1

                pose = airsim.Pose(
                    airsim.Vector3r(cx, cy, float(pos[2])),
                    airsim.Quaternionr(0, 0, 0, 1)
                )

                scale_ok = False
                try:
                    self.client.simSetObjectScale(name, airsim.Vector3r(size, size, height))
                    scale_ok = True
                except Exception as e:
                    print(f"  [ring {ring_idx} slot {slot_idx-1}] scale error: {e}")

                pose_ok = False
                try:
                    self.client.simSetObjectPose(name, pose, teleport=True)
                    pose_ok = True
                except Exception as e:
                    print(f"  [ring {ring_idx} slot {slot_idx-1}] pose error: {e}")

                if pose_ok:
                    actual_radius = (size * 0.6) if scale_ok else 0.8
                    self.obstacle_positions.append(pos.copy())
                    self.obstacle_radii.append(actual_radius)
                    placed_this_ring += 1
                    total_placed += 1
                    print(f"  [ring {ring_idx} r={ring_radius:.0f}m] "
                          f"obs {total_placed}: angle={math.degrees(angle):5.1f}° "
                          f"pos=({cx:.1f},{cy:.1f}) size={size:.2f}")

            status = ("OK" if placed_this_ring == obs_this_ring
                      else f"WARNING: only {placed_this_ring}/{obs_this_ring}")
            print(f"  Ring {ring_idx} (r={ring_radius:.0f}m): {status}")

        print(f"  [ObstacleMgr] total placed: {total_placed}/{num_obstacles}")

    def setup_goal_marker(self, position: np.ndarray) -> bool:
        if not self.goal_marker_exists:
            return False
        try:
            s = self.config.marker_size
            self.client.simSetObjectScale(self.GOAL_MARKER_NAME, airsim.Vector3r(s, s, s))
            self.client.simSetObjectPose(
                self.GOAL_MARKER_NAME,
                airsim.Pose(airsim.Vector3r(*position.tolist()), airsim.Quaternionr(0, 0, 0, 1)),
                teleport=True
            )
            return True
        except Exception:
            return False

    def check_obstacle_collision(self, drone_pos: np.ndarray) -> bool:
        for pos, radius in zip(self.obstacle_positions, self.obstacle_radii):
            if np.linalg.norm(drone_pos - pos) < radius + 0.5:
                return True
        return False

    def nearest_clearance(self, drone_pos: np.ndarray) -> float:
        if not self.obstacle_positions:
            return float('inf')
        return min(
            np.linalg.norm(drone_pos - p) - r
            for p, r in zip(self.obstacle_positions, self.obstacle_radii)
        )

    def destroy_all(self):
        for name in self.spawned_obstacles:
            try:
                self.client.simDestroyObject(name)
            except Exception:
                pass
        if self.goal_marker_exists:
            try:
                self.client.simSetObjectPose(
                    self.GOAL_MARKER_NAME,
                    airsim.Pose(self.HIDDEN_POS, airsim.Quaternionr(0, 0, 0, 1)),
                    teleport=True
                )
            except Exception:
                pass
        self.spawned_obstacles = []
        self.obstacle_positions = []
        self.obstacle_radii = []


# ---------------------------------------------------------------------------
# Image Processor
# ---------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.depth_buffer: deque = deque(maxlen=config.frame_stack)
        self._blank = np.zeros((config.image_height, config.image_width), dtype=np.float32)
        self._clear()

    def _clear(self):
        self.depth_buffer.clear()
        for _ in range(self.config.frame_stack):
            self.depth_buffer.append(self._blank.copy())

    def reset(self):
        self._clear()

    def process_and_push(self, raw: np.ndarray):
        cfg = self.config
        resized = cv2.resize(raw, (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_AREA)
        normalized = 1.0 - np.clip(resized / cfg.max_depth, 0.0, 1.0)
        self.depth_buffer.append(normalized.astype(np.float32))

    def get_stacked(self) -> np.ndarray:
        return np.array(list(self.depth_buffer), dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DroneObstacleEnv(gym.Env):
    """
    Fast goal-reaching drone environment with obstacle avoidance.
    """

    metadata = {"render_modes": ["depth_array"]}

    def __init__(self, config: EnvConfig = None):
        super().__init__()
        self.config = config or EnvConfig()
        cfg = self.config

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name="")
        self.client.armDisarm(True, vehicle_name="")

        self.obstacle_manager = ObstacleManager(self.client, cfg)
        self.image_processor = ImageProcessor(cfg)

        self.step_count = 0
        self.episode_reward = 0.0
        self.ep = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.goal_position = np.zeros(3, dtype=np.float64)
        self.drone_start = np.zeros(3, dtype=np.float64)
        self.curriculum_level = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self._vec_frame_dim = 12
        self._vec_history: deque = deque(maxlen=cfg.vec_stack)
        self._vec_blank = np.zeros(self._vec_frame_dim, dtype=np.float32)
        self._clear_vec_history()

        self.observation_space = spaces.Dict({
            "depth":       spaces.Box(0.0, 1.0,
                                      shape=(cfg.frame_stack, cfg.image_height, cfg.image_width),
                                      dtype=np.float32),
            "vec_hist":    spaces.Box(-np.inf, np.inf,
                                      shape=(cfg.vec_stack * self._vec_frame_dim,),
                                      dtype=np.float32),
            "prev_action": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        })

    def _clear_vec_history(self):
        self._vec_history.clear()
        for _ in range(self.config.vec_stack):
            self._vec_history.append(self._vec_blank.copy())

    def _push_vec_frame(self, goal_body, vel_body, omega_body, acc_body):
        frame = np.concatenate([goal_body, vel_body, omega_body, acc_body]).astype(np.float32)
        self._vec_history.append(frame)

    def _get_vec_hist(self) -> np.ndarray:
        return np.concatenate(list(self._vec_history), axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(np.clip(level, 0, len(CURRICULUM_LEVELS) - 1))

    def get_curriculum_params(self) -> Tuple[int, float]:
        """Returns (num_obstacles, reach_threshold_m)."""
        return CURRICULUM_LEVELS[self.curriculum_level]

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_kinematics(self):
        k = self.client.getMultirotorState(vehicle_name="").kinematics_estimated
        pos = np.array([k.position.x_val,         k.position.y_val,         k.position.z_val])
        ori = np.array([k.orientation.w_val,       k.orientation.x_val,
                        k.orientation.y_val,       k.orientation.z_val])
        vel = np.array([k.linear_velocity.x_val,   k.linear_velocity.y_val,
                        k.linear_velocity.z_val])
        acc = np.array([k.linear_acceleration.x_val, k.linear_acceleration.y_val,
                        k.linear_acceleration.z_val])
        return pos, ori, vel, acc

    def _get_imu_omega(self) -> np.ndarray:
        try:
            d = self.client.getImuData(imu_name='', vehicle_name='')
            return np.array([d.angular_velocity.x_val,
                              d.angular_velocity.y_val,
                              d.angular_velocity.z_val], dtype=np.float32)
        except Exception:
            k = self.client.getMultirotorState(vehicle_name="").kinematics_estimated
            av = k.angular_velocity
            return np.array([av.x_val, av.y_val, av.z_val], dtype=np.float32)

    def _get_depth_image(self) -> np.ndarray:
        cfg = self.config
        reqs = [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)]
        resp = self.client.simGetImages(reqs, vehicle_name="")
        if resp and resp[0].width > 0:
            d = airsim.list_to_2d_float_array(resp[0].image_data_float,
                                               resp[0].width, resp[0].height)
            return d.astype(np.float32)
        return np.zeros((cfg.image_height, cfg.image_width), dtype=np.float32)

    @staticmethod
    def _R(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
            [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
            [2*x*z - 2*w*y,       2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
        ])

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Tuple[dict, np.ndarray, np.ndarray]:
        pos, ori, vel_w, acc_w = self._get_kinematics()
        omega = self._get_imu_omega()
        R = self._R(ori)

        goal_body  = (R.T @ (self.goal_position - pos)).astype(np.float32)
        vel_body   = (R.T @ vel_w).astype(np.float32)
        acc_body   = (R.T @ acc_w).astype(np.float32)

        self._push_vec_frame(goal_body, vel_body, omega, acc_body)

        raw_depth = self._get_depth_image()
        self.image_processor.process_and_push(raw_depth)
        latest_depth = self.image_processor.depth_buffer[-1]

        goal_body_norm = float(np.linalg.norm(goal_body))
        goal_body_unit = goal_body / (goal_body_norm + 1e-6)

        return {
            "depth":       self.image_processor.get_stacked(),
            "vec_hist":    self._get_vec_hist(),
            "prev_action": self.prev_action.copy(),
        }, goal_body_unit, latest_depth

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray,
                        goal_body_unit: np.ndarray, latest_depth: np.ndarray,
                        collision: bool) -> Tuple[float, bool, dict]:
        cfg = self.config

        if collision:
            return cfg.collision_penalty, True, {"termination": "collision"}

        _, reach_threshold = self.get_curriculum_params()
        dist = float(np.linalg.norm(self.goal_position - pos))
        if dist < reach_threshold:
            return cfg.success_reward, True, {
                "termination": "success", "dist_to_goal": dist}

        if (pos[0] < cfg.arena_x_min or pos[0] > cfg.arena_x_max or
                pos[1] < cfg.arena_y_min or pos[1] > cfg.arena_y_max or
                pos[2] < cfg.arena_z_min or pos[2] > cfg.arena_z_max):
            return cfg.oob_penalty, True, {"termination": "out_of_bounds"}

        done = self.step_count >= cfg.max_steps
        info: dict = {"termination": "timeout"} if done else {}

        goal_dir = (self.goal_position - pos) / (dist + 1e-6)

        d_dot = float(np.dot(vel, goal_dir))
        brake = float(np.tanh(dist / (cfg.d_brake + 1e-6)))
        r_approach = cfg.w_approach * float(np.clip(d_dot, -15.0, 15.0)) * brake

        d_dot_dot = float(np.dot(acc, goal_dir))
        r_accel = cfg.w_accel * float(np.clip(d_dot_dot, -15.0, 15.0)) * brake

        r_dist = cfg.w_dist * float(math.exp(-cfg.k_dist * dist * dist))

        r_heading = cfg.w_heading * float(goal_body_unit[0])

        z_err = np.clip(goal_dir[2], -5, 5)
        r_alt = -5 * z_err**2

        max_depth_val = float(np.max(latest_depth))
        if max_depth_val > cfg.depth_danger_thresh:
            excess = (max_depth_val - cfg.depth_danger_thresh) / (1.0 - cfg.depth_danger_thresh + 1e-6)
            r_depth = -cfg.w_depth_penalty * (excess ** cfg.depth_penalty_sharpness)
        else:
            r_depth = 0.0

        reward = r_approach + r_accel + r_dist + r_alt + r_heading + r_depth + cfg.time_penalty

        info.update({
            "dist_to_goal":          dist,
            "brake":                 brake,
            "d_dot":                 d_dot,
            "d_dot_dot":             d_dot_dot,
            "r_approach":            r_approach,
            "r_accel":               r_accel,
            "r_dist":                r_dist,
            "r_heading":             r_heading,
            "r_depth":               r_depth,
            "max_depth_val":         max_depth_val,
            "nearest_obs_clearance": self.obstacle_manager.nearest_clearance(pos),
        })
        return reward, done, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        self.step_count += 1
        cfg = self.config

        roll     = float(action[0]) * cfg.max_roll
        pitch    = float(action[1]) * cfg.max_pitch
        yaw_rate = float(action[2]) * cfg.max_yaw_rate
        throttle = (float(action[3]) + 1.0) / 2.0 * cfg.max_throttle

        self.client.moveByRollPitchYawrateThrottleAsync(
            roll, pitch, yaw_rate, throttle, cfg.dt, vehicle_name=""
        ).join()

        self.prev_action = action.astype(np.float32)

        pos, ori, vel, acc = self._get_kinematics()

        collision = False
        try:
            collision = self.client.simGetCollisionInfo(vehicle_name="").has_collided
        except Exception:
            pass
        collision = collision or self.obstacle_manager.check_obstacle_collision(pos)

        obs, goal_body_unit, latest_depth = self._get_obs()
        reward, done, info = self._compute_reward(
            pos, vel, acc, goal_body_unit, latest_depth, collision)
        self.episode_reward += reward

        self.client.simPlotPoints(
            [airsim.Vector3r(float(self.goal_position[0]),
                             float(self.goal_position[1]),
                             float(self.goal_position[2]))],
            [1.0, 0.5, 0.0, 1.0], 10.0, 0.1, False
        )

        info["step"] = self.step_count
        info["episode_reward"] = self.episode_reward
        info["position"] = pos.tolist()
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        cfg = self.config
        self.ep += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)

        self.client.reset()
        time.sleep(0.1)
        self.client.enableApiControl(True, vehicle_name="")
        self.client.armDisarm(True, vehicle_name="")
        self.image_processor.reset()
        self._clear_vec_history()

        num_obs, reach_threshold = self.get_curriculum_params()

        # ── Compute active ring geometry ──────────────────────────────────
        # Determine which ring radii are active given the obstacle count.
        num_rings    = max(1, math.ceil(num_obs / _OBS_PER_RING)) if num_obs > 0 else 0
        outer_radius = cfg.obstacle_ring_radii[min(num_rings, len(cfg.obstacle_ring_radii)) - 1] \
                       if num_rings > 0 else 0.0

        # ── Goal placement ────────────────────────────────────────────────
        # Goal must be far enough from the -X wall to fit the spawn distance.
        spawn_dist    = outer_radius + np.random.uniform(cfg.spawn_offset_min, cfg.spawn_offset_max)
        goal_x_min    = cfg.arena_x_min + spawn_dist + 3.0   # 3 m wall clearance
        goal_x_max    = cfg.arena_x_max - 3.0
        goal_x        = float(np.random.uniform(goal_x_min, goal_x_max))
        goal_y        = float(np.random.uniform(cfg.arena_y_min + 5, cfg.arena_y_max - 5))
        goal_z        = float(np.random.uniform(cfg.arena_z_min + 3, cfg.arena_z_max - 5))
        self.goal_position = np.array([goal_x, goal_y, goal_z])

        # ── Drone spawn ───────────────────────────────────────────────────
        # Placed (outer_radius + spawn_offset) metres behind the goal along -X.
        # A small Y jitter ensures the drone does not always approach dead-centre.
        drone_x = goal_x - spawn_dist
        drone_y = goal_y + float(np.random.uniform(-3.0, 3.0))
        drone_z = float(np.random.uniform(cfg.arena_z_min + 2, cfg.arena_z_max - 5))
        drone_x = float(np.clip(drone_x, cfg.arena_x_min + 3, cfg.arena_x_max - 3))
        drone_y = float(np.clip(drone_y, cfg.arena_y_min + 3, cfg.arena_y_max - 3))
        self.drone_start = np.array([drone_x, drone_y, drone_z])

        # Teleport + takeoff
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(*self.drone_start.tolist()), to_quaternion(0, 0, 0)),
            True, vehicle_name=""
        )
        time.sleep(0.1)
        self.client.takeoffAsync(vehicle_name="").join()

        # Setup obstacles and goal marker
        self.obstacle_manager.setup_episode(num_obs, self.drone_start, self.goal_position)
        self.obstacle_manager.setup_goal_marker(self.goal_position)

        # Prime the depth buffer
        for _ in range(cfg.frame_stack):
            self.image_processor.process_and_push(self._get_depth_image())

        obs, _, _ = self._get_obs()

        episode_dist = float(np.linalg.norm(self.goal_position - self.drone_start))
        info = {
            "ep":               self.ep,
            "curriculum_level": self.curriculum_level,
            "num_obstacles":    num_obs,
            "num_rings":        num_rings,
            "reach_threshold":  reach_threshold,
            "drone_start":      self.drone_start.tolist(),
            "goal_position":    self.goal_position.tolist(),
            "episode_dist":     episode_dist,
            "spawn_dist":       float(spawn_dist),
        }
        print(f"  Ep {self.ep} | level={self.curriculum_level} "
              f"rings={num_rings} obs={num_obs} thresh={reach_threshold:.1f}m "
              f"spawn={spawn_dist:.1f}m dist={episode_dist:.1f}m")
        return obs, info

    def render(self, mode="depth_array"):
        return self._get_depth_image() if mode == "depth_array" else None

    def close(self):
        self.obstacle_manager.destroy_all()
        self.client.armDisarm(False, vehicle_name="")
        self.client.enableApiControl(False, vehicle_name="")
        print("Environment closed.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Curriculum levels:")
    for i, (n, t) in enumerate(CURRICULUM_LEVELS):
        num_rings = max(1, math.ceil(n / _OBS_PER_RING)) if n > 0 else 0
        print(f"  Level {i:2d}: {n:2d} obstacle(s), {num_rings} ring(s), {t:.1f} m threshold")