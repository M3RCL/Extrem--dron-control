#!/usr/bin/env python3
"""
Training Script for Improved GPS-Based Drone Environment
Uses realistic sensors with better episode management
"""

import numpy as np
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import argparse
from improved_drone_env import ImprovedDroneEnv
import torch as th
import json
from datetime import datetime


class EpisodeStatsCallback(BaseCallback):
    """
    Custom callback to log episode statistics
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_stats_list = []
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones'):
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'episode_stats' in info and info['episode_stats']:
                        stats = info['episode_stats'].copy()
                        stats['episode_num'] = len(self.episode_stats_list)
                        stats['total_steps'] = self.num_timesteps
                        self.episode_stats_list.append(stats)
                        
                        # Log to console
                        if self.verbose > 0:
                            term_reason = stats.get('termination_reason', 'unknown')
                            max_alt = stats.get('max_altitude', 0)
                            collision = stats.get('collision_detected', False)
                            print(f"Episode {len(self.episode_stats_list)}: "
                                  f"Reason={term_reason}, MaxAlt={max_alt:.2f}m, "
                                  f"Collision={collision}")
        
        return True
    
    def save_stats(self, filepath):
        """Save episode statistics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.episode_stats_list, f, indent=2)


def make_env(target_position=None, randomize_target=False, max_steps=500):
    """Create and wrap environment"""
    def _init():
        env = ImprovedDroneEnv(
            target_position=target_position,
            max_steps=max_steps,
            max_distance_from_target=20.0,
            min_altitude=0.1,
            max_altitude=10.0,
            max_tilt_angle=60.0
        )
        env = Monitor(env)
        return env
    return _init


def train_drone_agent(
    algorithm='SAC',
    total_timesteps=1_000_000,
    save_dir='./models',
    log_dir='./logs',
    eval_freq=10000,
    checkpoint_freq=50000,
    randomize_target=False,
    normalize_observations=True
):
    """
    Train RL agent for drone control
    
    Args:
        algorithm: 'SAC', 'PPO', or 'TD3'
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation (timesteps)
        checkpoint_freq: Frequency of checkpointing (timesteps)
        randomize_target: Randomize target position each episode
        normalize_observations: Use observation normalization
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{timestamp}"
    
    print("="*60)
    print(f"Training Configuration: {run_name}")
    print("="*60)
    print(f"Algorithm: {algorithm}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Randomize target: {randomize_target}")
    print(f"Normalize observations: {normalize_observations}")
    print(f"Using GPS for positioning (no ground truth!)")
    print("="*60)
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = DummyVecEnv([make_env(
        target_position=None if randomize_target else np.array([5., 0., 2.]),
        randomize_target=randomize_target,
        max_steps=500
    )])
    
    # Normalize observations (helps with GPS noise)
    if normalize_observations:
        print("Applying observation normalization...")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(
        target_position=np.array([5., 0., 2.]),
        max_steps=1000
    )])
    
    if normalize_observations:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize reward during eval
            clip_obs=10.0,
            training=False
        )
    
    # Create model
    print(f"\nInitializing {algorithm} agent...")
    
    if algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=th.nn.ReLU,
            ),
            verbose=1,
            tensorboard_log=log_dir,
        )
    
    elif algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=th.nn.ReLU,
            ),
            verbose=1,
            tensorboard_log=log_dir,
        )
    
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            train_env,
            learning_rate=1e-3,
            buffer_size=200000,
            learning_starts=10000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, 'episode'),
            gradient_steps=-1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=dict(
                net_arch=[400, 300],
                activation_fn=th.nn.ReLU,
            ),
            verbose=1,
            tensorboard_log=log_dir,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    episode_stats_callback = EpisodeStatsCallback(verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=f'{run_name}_checkpoint'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_dir}/{run_name}_best',
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    print("\nStarting training...")
    print(f"Monitor progress: tensorboard --logdir {log_dir}")
    print("-"*60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[episode_stats_callback, checkpoint_callback, eval_callback],
            log_interval=10,
            tb_log_name=run_name,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_dir, f'{run_name}_final')
        model.save(final_path)
        
        # Save normalization stats if used
        if normalize_observations:
            train_env.save(os.path.join(save_dir, f'{run_name}_vecnormalize.pkl'))
        
        # Save episode statistics
        stats_path = os.path.join(save_dir, f'{run_name}_episode_stats.json')
        episode_stats_callback.save_stats(stats_path)
        
        print("\n" + "="*60)
        print(f"Training complete!")
        print(f"Final model saved: {final_path}")
        print(f"Episode stats saved: {stats_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        interrupted_path = os.path.join(save_dir, f'{run_name}_interrupted')
        model.save(interrupted_path)
        if normalize_observations:
            train_env.save(os.path.join(save_dir, f'{run_name}_vecnormalize_interrupted.pkl'))
        print(f"Model saved to {interrupted_path}")
    
    finally:
        train_env.close()
        eval_env.close()


def test_agent(model_path, num_episodes=10, normalize_path=None):
    """
    Test trained agent
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of test episodes
        normalize_path: Path to VecNormalize stats (if used during training)
    """
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path)  # Change based on your algorithm
    
    # Create test environment
    env = DummyVecEnv([make_env(
        target_position=np.array([5., 0., 2.]),
        max_steps=1000
    )])
    
    # Load normalization if used
    if normalize_path and os.path.exists(normalize_path):
        print(f"Loading normalization from {normalize_path}")
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Test episodes
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*60)
        
        while steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            steps += 1
            
            if steps % 50 == 0:
                dist = info[0].get('distance_to_target', 0)
                alt = info[0].get('altitude', 0)
                print(f"  Step {steps}: distance={dist:.2f}m, altitude={alt:.2f}m")
            
            if done[0]:
                term_reason = info[0].get('termination_reason', 'unknown')
                print(f"\nTerminated: {term_reason}")
                print(f"Total steps: {steps}")
                print(f"Episode reward: {episode_reward:.2f}")
                
                if term_reason == 'success':
                    success_count += 1
                    print("✓ SUCCESS!")
                elif 'collision' in term_reason or 'tilt' in term_reason:
                    collision_count += 1
                    print("✗ Collision/Crash")
                
                if 'episode_stats' in info[0]:
                    stats = info[0]['episode_stats']
                    print(f"Max altitude: {stats.get('max_altitude', 0):.2f}m")
                    print(f"Max distance: {stats.get('max_distance', 0):.2f}m")
                
                break
    
    env.close()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Collision rate: {collision_count}/{num_episodes} ({100*collision_count/num_episodes:.1f}%)")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Test improved drone RL agent')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--algorithm', choices=['SAC', 'PPO', 'TD3'], default='SAC')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--model_path', type=str, default='./models/SAC_*_final.zip')
    parser.add_argument('--normalize_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--randomize_target', action='store_true')
    parser.add_argument('--no_normalize', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_drone_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            randomize_target=args.randomize_target,
            normalize_observations=not args.no_normalize
        )
    else:
        test_agent(args.model_path, num_episodes=10, normalize_path=args.normalize_path)