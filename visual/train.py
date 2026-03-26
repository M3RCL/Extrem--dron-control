"""
Training Script — Drone Obstacle Avoidance SAC
================================================
Usage:
  python train.py train
  python train.py train --resume
  python train.py eval --checkpoint checkpoints/latest.pth
"""

import os
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from drone_obstacle_env import DroneObstacleEnv, EnvConfig, CURRICULUM_LEVELS
from sac_multitask import SAC, SACConfig, ReplayBuffer


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    max_episodes:   int   = 15_000
    warmup_steps:   int   = 1_0    # random actions before learning starts
    train_freq:     int   = 1        # gradient steps per environment step
    save_freq:      int   = 100      # episodes between checkpoints
    log_freq:       int   = 10       # episodes between console prints

    # Curriculum advancement
    curriculum_window:    int   = 50    # look-back window for success rate
    curriculum_threshold: float = 0.40  # success rate required to advance


# ---------------------------------------------------------------------------
# Curriculum controller
# ---------------------------------------------------------------------------

class Curriculum:
    """Advances a single integer level when success rate exceeds a threshold."""

    def __init__(self, cfg: TrainConfig):
        self.cfg     = cfg
        self.level   = 0
        self.max     = len(CURRICULUM_LEVELS) - 1
        self.results = []   # rolling window of bool successes

    def record(self, success: bool) -> bool:
        """Record episode outcome. Returns True if level just advanced."""
        self.results.append(float(success))
        if len(self.results) < self.cfg.curriculum_window:
            return False
        # Keep only the last window
        self.results = self.results[-self.cfg.curriculum_window:]
        rate = np.mean(self.results)
        if rate >= self.cfg.curriculum_threshold and self.level < self.max:
            self.level += 1
            self.results = []   # reset window after advancing
            n_obs, thresh = CURRICULUM_LEVELS[self.level]
            print(f"\n{'='*60}")
            print(f"  CURRICULUM ADVANCE → level {self.level}")
            print(f"  {n_obs} obstacle(s), {thresh:.1f} m threshold")
            print(f"{'='*60}\n")
            return True
        return False

    def success_rate(self) -> float:
        return float(np.mean(self.results)) if self.results else 0.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    HEADER = ("ep,steps,reward,success,termination,level,"
              "num_obs,thresh,dist_to_goal,nearest_obs,"
              "critic_loss,actor_loss,alpha,q_mean\n")

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"train_{ts}.csv")
        with open(self.path, "w") as f:
            f.write(self.HEADER)

    def log(self, ep: int, steps: int, reward: float, success: bool,
            termination: str, level: int, num_obs: int, thresh: float,
            dist: float, nearest_obs: float, metrics: dict):
        with open(self.path, "a") as f:
            f.write(
                f"{ep},{steps},{reward:.2f},{int(success)},{termination},"
                f"{level},{num_obs},{thresh:.2f},{dist:.2f},{nearest_obs:.2f},"
                f"{metrics.get('critic_loss', 0):.4f},"
                f"{metrics.get('actor_loss',  0):.4f},"
                f"{metrics.get('alpha',       0):.4f},"
                f"{metrics.get('q_mean',      0):.3f}\n"
            )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    env_cfg   = EnvConfig(dt=0.02, max_steps=args.max_steps)
    sac_cfg   = SACConfig(batch_size=args.batch_size,
                          buffer_capacity=args.buffer_size)
    train_cfg = TrainConfig(max_episodes=args.max_episodes)

    print("Initialising environment …")
    env = DroneObstacleEnv(env_cfg)

    print("Initialising SAC agent …")
    agent = SAC(
        frame_stack=env_cfg.frame_stack,
        image_size=env_cfg.image_height,
        action_dim=4,
        vec_hist_dim=env_cfg.vec_stack * 12,   # vec_stack * (goal+vel+omega+acc = 12)
        config=sac_cfg,
    )

    # Buffer
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        depth_shape=(env_cfg.frame_stack, env_cfg.image_height, env_cfg.image_width),
        state_vec_dim=agent.state_vec_dim,
        action_dim=4,
    )

    # Checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pth")
    start_ep  = 0
    if args.resume and os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        start_step = agent.train_steps   # rough proxy; not episode count
        start_ep = start_ep//1000

    curriculum = Curriculum(train_cfg)
    logger     = Logger(args.log_dir)

    total_steps = 0
    metrics:    dict = {}

    print(f"\nTraining for {args.max_episodes} episodes")
    print(f"  Warmup: {train_cfg.warmup_steps} steps")
    print(f"  Curriculum levels: {len(CURRICULUM_LEVELS)}  "
          f"(advance when success rate ≥ {train_cfg.curriculum_threshold:.0%} "
          f"over {train_cfg.curriculum_window} eps)\n")

    for ep in range(start_ep, args.max_episodes):
        env.set_curriculum_level(curriculum.level)
        num_obs, thresh = env.get_curriculum_params()

        obs, _ = env.reset()
        ep_reward   = 0.0
        ep_dist     = float(np.linalg.norm(
            np.array(env.goal_position) - np.array(env.drone_start)))
        min_nearest = float('inf')
        last_info   = {}

        for step in range(args.max_steps):
            total_steps += 1

            if total_steps < train_cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, _, info = env.step(action)
            ep_reward += reward

            # Track nearest obstacle clearance over episode
            nc = info.get("nearest_obs_clearance", float('inf'))
            if nc < min_nearest:
                min_nearest = nc

            # Store transition
            sv      = agent.obs_to_state_vec(obs)
            next_sv = agent.obs_to_state_vec(next_obs)
            buffer.add(
                obs["depth"], sv, action, reward,
                next_obs["depth"], next_sv, done
            )

            # Gradient update
            if (total_steps >= train_cfg.warmup_steps
                    and total_steps % train_cfg.train_freq == 0
                    and len(buffer) >= sac_cfg.batch_size):
                metrics = agent.train(buffer)

            obs = next_obs
            last_info = info
            if done:
                print(
                f"Ep {ep:5d} | R={ep_reward:7.1f} | "
                )
                break

        # Episode bookkeeping
        termination = last_info.get("termination", "unknown")
        success     = termination == "success"
        advanced    = curriculum.record(success)
        dist_final  = last_info.get("dist_to_goal", ep_dist)

        logger.log(ep, step + 1, ep_reward, success, termination,
                   curriculum.level, num_obs, thresh,
                   dist_final, min_nearest, metrics)

        if ep % train_cfg.log_freq == 0:
            print(
                f"Ep {ep:5d} | R={ep_reward:7.1f} | "
                f"{'✓' if success else '✗'} {termination:<15} | "
                f"lvl={curriculum.level} obs={num_obs} th={thresh:.1f}m | "
                f"sr={curriculum.success_rate():.2f} | "
                f"dist={dist_final:.1f}m nc={min_nearest:.1f}m | "
                f"α={metrics.get('alpha', 0):.3f} "
                f"cL={metrics.get('critic_loss', 0):.3f}"
            )

        if ep % train_cfg.save_freq == 0 and ep > 0:
            agent.save(ckpt_path)

    agent.save(ckpt_path)
    env.close()
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    env_cfg = EnvConfig(dt=0.02, max_steps=args.max_steps)
    env     = DroneObstacleEnv(env_cfg)
    agent   = SAC(frame_stack=env_cfg.frame_stack, image_size=env_cfg.image_height,
                  action_dim=4, vec_hist_dim=env_cfg.vec_stack * 12)
    agent.load(args.checkpoint)

    results = {t: 0 for t in ["success", "collision", "timeout", "out_of_bounds", "unknown"]}

    for level_idx in range(len(CURRICULUM_LEVELS)):
        env.set_curriculum_level(level_idx)
        n_obs, thresh = env.get_curriculum_params()
        print(f"\nLevel {level_idx}: {n_obs} obstacles, {thresh:.1f} m threshold")

        ep_rewards = []
        for _ in range(args.eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                action = agent.select_action(obs, deterministic=True)
                obs, r, done, _, info = env.step(action)
                ep_r += r
            ep_rewards.append(ep_r)
            t = info.get("termination", "unknown")
            results[t] = results.get(t, 0) + 1

        print(f"  Mean reward: {np.mean(ep_rewards):.1f}  ±{np.std(ep_rewards):.1f}")

    total = sum(results.values())
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:<20} {v:4d}  ({v/total*100:.1f}%)")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train")
    t.add_argument("--max-episodes",   type=int,   default=50_000)
    t.add_argument("--max-steps",      type=int,   default=1_000)
    t.add_argument("--batch-size",     type=int,   default=256)
    t.add_argument("--buffer-size",    type=int,   default=100_000)
    t.add_argument("--checkpoint-dir", default="checkpoints")
    t.add_argument("--log-dir",        default="logs")
    t.add_argument("--resume",         action="store_true", default=True)

    # eval
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint",    required=True)
    e.add_argument("--max-steps",     type=int, default=1_000)
    e.add_argument("--eval-episodes", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        parser.print_help()