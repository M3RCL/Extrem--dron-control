"""
SAC Agent for Drone Obstacle Avoidance
========================================
Architecture:
  CNN encoder  : (frame_stack, 64, 64) → 256-dim feature
  State vector : [cnn_feat(256) | goal_body(3) | vel_body(3) | omega_body(3) | prev_action(4)] = 269
  Actor        : MLP, Gaussian policy with tanh squashing
  Critic       : twin Q-networks (standard SAC)
  Buffer       : flat replay buffer with uint8 depth storage (memory efficient)

No LSTM, no pose estimation, no multi-task critics.
The depth stack already provides temporal context.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SAC using device: {device}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    # CNN
    cnn_channels: Tuple[int, ...] = (32, 64, 64)
    cnn_kernels:  Tuple[int, ...] = (8, 4, 3)
    cnn_strides:  Tuple[int, ...] = (4, 2, 1)
    cnn_feat_dim: int = 256

    # Networks
    hidden_dim: int = 256

    # SAC hyper-parameters
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    encoder_lr: float = 1e-4
    init_temperature: float = 0.2
    batch_size: int = 256

    # Buffer
    buffer_capacity: int = 100_000


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    Encodes a stacked depth image (frame_stack, H, W) → feat_dim vector.

    Channels: 32 → 64 → 64
    Kernels / strides chosen for 64×64 input:
      64 → 15 → 6 → 4  after the three conv layers
    Flat: 64*4*4 = 1024 → Linear → feat_dim
    """

    def __init__(self, frame_stack: int = 4, image_size: int = 64,
                 channels=(32, 64, 64), kernels=(8, 4, 3), strides=(4, 2, 1),
                 feat_dim: int = 256):
        super().__init__()
        layers = []
        in_ch = frame_stack
        for out_ch, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv2d(in_ch, out_ch, k, s), nn.ReLU()]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, image_size, image_size)
            flat = self.cnn(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Tanh(),
        )
        self.feat_dim = feat_dim
        print(f"CNNEncoder: flat={flat} → feat={feat_dim}")

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (B, C, H, W)  →  (B, feat_dim)"""
        x = self.cnn(depth)
        return self.fc(x.view(depth.size(0), -1))


# ---------------------------------------------------------------------------
# Actor (MLP, no LSTM)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),   nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),   nn.ReLU(),
            nn.Linear(hidden, 128),nn.LayerNorm(128), nn.ReLU(),
        )
        self.mean    = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean    = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean), torch.zeros(mean.size(0), 1, device=mean.device)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x    = dist.rsample()
        a    = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, log_prob


# ---------------------------------------------------------------------------
# Twin-Q Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        inp = state_dim + action_dim

        def _mlp():
            return nn.Sequential(
                nn.Linear(inp,    hidden),nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden),nn.ReLU(),
                nn.Linear(hidden, hidden),nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, 128),nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# Flat Replay Buffer  (uint8 depth storage to save ~4× RAM)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores depth as uint8 ([0,255]) and converts to float32 on sampling.
    For 100k transitions with depth (4,64,64):
      uint8  ≈  1.6 GB
      float32 ≈  6.4 GB
    All other arrays are float32.
    """

    def __init__(self, capacity: int, depth_shape: Tuple,
                 state_vec_dim: int, action_dim: int):
        self.capacity      = capacity
        self.depth_shape   = depth_shape        # e.g. (4, 64, 64)
        self.state_vec_dim = state_vec_dim
        self.action_dim    = action_dim
        self.ptr           = 0
        self.size          = 0

        # Depth stored as uint8 [0,255]; multiply by (1/255) on load
        self.depth      = np.zeros((capacity, *depth_shape), dtype=np.uint8)
        self.next_depth = np.zeros((capacity, *depth_shape), dtype=np.uint8)

        # Everything else as float32
        self.state_vec      = np.zeros((capacity, state_vec_dim), dtype=np.float32)
        self.next_state_vec = np.zeros((capacity, state_vec_dim), dtype=np.float32)
        self.actions        = np.zeros((capacity, action_dim),    dtype=np.float32)
        self.rewards        = np.zeros((capacity, 1),             dtype=np.float32)
        self.dones          = np.zeros((capacity, 1),             dtype=np.float32)

    @staticmethod
    def _to_uint8(depth: np.ndarray) -> np.ndarray:
        return (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    def add(self, depth: np.ndarray, state_vec: np.ndarray, action: np.ndarray,
            reward: float, next_depth: np.ndarray, next_state_vec: np.ndarray,
            done: bool):
        i = self.ptr
        self.depth[i]           = self._to_uint8(depth)
        self.next_depth[i]      = self._to_uint8(next_depth)
        self.state_vec[i]       = state_vec
        self.next_state_vec[i]  = next_state_vec
        self.actions[i]         = action
        self.rewards[i]         = reward
        self.dones[i]           = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        def t(x): return torch.FloatTensor(x).to(device)

        depth      = t(self.depth[idx].astype(np.float32)      / 255.0)
        next_depth = t(self.next_depth[idx].astype(np.float32) / 255.0)

        return {
            "depth":          depth,
            "state_vec":      t(self.state_vec[idx]),
            "next_depth":     next_depth,
            "next_state_vec": t(self.next_state_vec[idx]),
            "actions":        t(self.actions[idx]),
            "rewards":        t(self.rewards[idx]),
            "dones":          t(self.dones[idx]),
        }

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SAC:
    """
    Standard Soft Actor-Critic with:
      - CNN encoder shared between actor and critic
      - Separate encoder target for critic stability
      - Automatic entropy tuning
    """

    def __init__(self, frame_stack: int = 4, image_size: int = 64,
                 action_dim: int = 4,
                 vec_hist_dim: int = 36,   # vec_stack(3) * 12  from env
                 config: SACConfig = None):
        self.cfg = config or SACConfig()
        cfg = self.cfg

        # Encoder
        self.encoder = CNNEncoder(
            frame_stack, image_size,
            cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides, cfg.cnn_feat_dim
        ).to(device)
        self.encoder_target = CNNEncoder(
            frame_stack, image_size,
            cfg.cnn_channels, cfg.cnn_kernels, cfg.cnn_strides, cfg.cnn_feat_dim
        ).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # State vec = vec_hist (stacked goal/vel/omega/acc) + prev_action
        self.state_vec_dim = vec_hist_dim + action_dim
        state_dim = cfg.cnn_feat_dim + self.state_vec_dim

        self.actor  = Actor(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature
        self.log_alpha    = torch.tensor(
            np.log(cfg.init_temperature), device=device, requires_grad=True)
        self.target_entropy = -float(action_dim)

        # Optimizers
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)
        self.actor_opt   = optim.Adam(self.actor.parameters(),   lr=cfg.actor_lr)
        self.critic_opt  = optim.Adam(self.critic.parameters(),  lr=cfg.critic_lr)
        self.alpha_opt   = optim.Adam([self.log_alpha],          lr=cfg.alpha_lr)

        self.train_steps = 0

        total_params = sum(p.numel() for p in list(self.encoder.parameters()) +
                           list(self.actor.parameters()) + list(self.critic.parameters()))
        print(f"SAC: state_dim={state_dim}, state_vec_dim={self.state_vec_dim}, "
              f"total params={total_params:,}")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Observation → tensors
    # ------------------------------------------------------------------

    def _obs_to_tensors(self, obs: dict):
        depth     = torch.FloatTensor(obs["depth"]).unsqueeze(0).to(device)
        state_vec = torch.FloatTensor(
            np.concatenate([obs["vec_hist"], obs["prev_action"]])
        ).unsqueeze(0).to(device)
        return depth, state_vec

    @staticmethod
    def _build_state(enc_feat: torch.Tensor, state_vec: torch.Tensor) -> torch.Tensor:
        return torch.cat([enc_feat, state_vec], dim=-1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        depth, state_vec = self._obs_to_tensors(obs)
        feat  = self.encoder(depth)
        state = self._build_state(feat, state_vec)
        action, _ = self.actor.sample(state, deterministic)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Build flat state vector from obs dict (for buffer storage)
    # ------------------------------------------------------------------

    @staticmethod
    def obs_to_state_vec(obs: dict) -> np.ndarray:
        return np.concatenate([
            obs["vec_hist"], obs["prev_action"]
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train(self, buffer: ReplayBuffer) -> dict:
        if len(buffer) < self.cfg.batch_size:
            return {}
        self.train_steps += 1
        cfg = self.cfg
        batch = buffer.sample(cfg.batch_size)

        depth          = batch["depth"]
        state_vec      = batch["state_vec"]
        next_depth     = batch["next_depth"]
        next_state_vec = batch["next_state_vec"]
        actions        = batch["actions"]
        rewards        = batch["rewards"]
        dones          = batch["dones"]

        # ── 1. Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            next_feat       = self.encoder_target(next_depth)
            next_state      = self._build_state(next_feat, next_state_vec)
            next_act, nlp   = self.actor.sample(next_state)
            nq1, nq2        = self.critic_target(next_state, next_act)
            target_q        = torch.min(nq1, nq2) - self.alpha * nlp
            target_q        = rewards + (1.0 - dones) * cfg.gamma * target_q
            # Guard against NaN from target networks
            target_q        = torch.nan_to_num(target_q, nan=0.0, posinf=500.0, neginf=-500.0)

        feat  = self.encoder(depth)
        state = self._build_state(feat, state_vec)
        q1, q2 = self.critic(state, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.encoder_opt.step()
        self.critic_opt.step()

        # ── 2. Actor update ───────────────────────────────────────────────
        feat  = self.encoder(depth)
        state = self._build_state(feat.detach(), state_vec)   # stop critic grad from actor pass
        new_act, log_prob = self.actor.sample(state)
        q1_pi = self.critic.q1_only(state, new_act)
        actor_loss = (self.alpha.detach() * log_prob - q1_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── 3. Temperature update ─────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── 4. Soft update targets ────────────────────────────────────────
        tau = cfg.tau
        for p, pt in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "q_mean":      ((q1 + q2) / 2).mean().item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "encoder":        self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "log_alpha":      self.log_alpha.data,
            "train_steps":    self.train_steps,
            "cfg":            self.cfg,
        }, path)
        print(f"  Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.encoder_target.load_state_dict(ckpt["encoder_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.train_steps = ckpt.get("train_steps", 0)
        print(f"  Loaded ← {path}  (step {self.train_steps})")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SAC smoke test ===")
    cfg = SACConfig(buffer_capacity=1000, batch_size=16)
    agent = SAC(frame_stack=4, image_size=64, action_dim=4, config=cfg)

    # vec_hist = vec_stack(3) * 12 = 36 dims
    dummy_obs = {
        "depth":       np.random.rand(4, 64, 64).astype(np.float32),
        "vec_hist":    np.random.randn(36).astype(np.float32),
        "prev_action": np.zeros(4,                  np.float32),
    }

    action = agent.select_action(dummy_obs)
    print(f"Action: {action}")

    buf = ReplayBuffer(1000, (4, 64, 64), agent.state_vec_dim, 4)
    for i in range(32):
        sv = agent.obs_to_state_vec(dummy_obs)
        buf.add(dummy_obs["depth"], sv, action, np.random.randn(),
                dummy_obs["depth"], sv, i == 31)

    print(f"Buffer size: {len(buf)}")
    m = agent.train(buf)
    print(f"Train metrics: {m}")
    agent.save("/tmp/test_sac.pth")
    agent.load("/tmp/test_sac.pth")
    print("All tests passed.")