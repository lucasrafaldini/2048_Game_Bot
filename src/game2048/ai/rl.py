"""Reinforcement Learning agents for 2048 (optional dependencies)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseAgent
from ..core.board import Board2048
from ..core.constants import Direction


class RLAgent(BaseAgent):
    """
    Base class for RL agents.
    
    Requires: torch, gymnasium, stable-baselines3
    Install with: pip install game2048[rl]
    """
    
    def __init__(self, model_path: str, name: str | None = None):
        super().__init__(name or "RL")
        self.model_path = model_path
        self._model = None
        self._env = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            import torch
            from stable_baselines3 import DQN, PPO
            import gymnasium as gym
            
            # Try to load as DQN first, then PPO
            for model_cls in [DQN, PPO]:
                try:
                    self._model = model_cls.load(self.model_path)
                    break
                except Exception:
                    continue
            
            if self._model is None:
                raise ValueError(f"Could not load model from {self.model_path}")
                
        except ImportError:
            raise ImportError(
                "RL dependencies not installed. "
                "Install with: pip install game2048[rl]"
            )
    
    def get_move(self, board: Board2048) -> Direction:
        if self._model is None:
            # Fallback to random
            valid = board.get_valid_moves()
            return valid[0] if valid else Direction.UP
        
        # Convert board to observation
        obs = self._board_to_obs(board)
        
        # Get action from model
        action, _ = self._model.predict(obs, deterministic=True)
        
        # Map action to direction
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        return directions[int(action)]
    
    def _board_to_obs(self, board: Board2048) -> np.ndarray:
        """Convert board to model observation format."""
        # Standard: 4x4x16 one-hot encoding (log2 of tile values)
        grid = board.to_numpy()
        obs = np.zeros((4, 4, 16), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                val = grid[r, c]
                if val > 0:
                    idx = int(np.log2(val))
                    if idx < 16:
                        obs[r, c, idx] = 1.0
        return obs.flatten()


class DQNAgent(RLAgent):
    """DQN-based agent."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, "DQN")


class PPOAgent(RLAgent):
    """PPO-based agent."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path, "PPO")


def create_random_rl_agent() -> "RandomRLAgent":
    """Create a random agent that mimics RL interface (for testing)."""
    return RandomRLAgent()


class RandomRLAgent(BaseAgent):
    """Random agent with RL interface for testing without dependencies."""
    
    def __init__(self):
        super().__init__("RandomRL")
    
    def get_move(self, board: Board2048) -> Direction:
        valid = board.get_valid_moves()
        import random
        return random.choice(valid) if valid else Direction.UP


# Training utilities (require full RL stack)
def train_dqn(
    total_timesteps: int = 1_000_000,
    save_path: str = "models/dqn_2048",
    **kwargs,
) -> None:
    """
    Train a DQN agent on 2048.
    
    Requires: pip install game2048[rl]
    """
    try:
        import gymnasium as gym
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise ImportError("Install RL dependencies: pip install game2048[rl]")
    
    # Import the custom environment
    from ..env import Game2048Env
    
    # Create environment
    env = Monitor(Game2048Env())
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        **kwargs,
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    model.save(save_path)
    print(f"Model saved to {save_path}")


def train_ppo(
    total_timesteps: int = 2_000_000,
    save_path: str = "models/ppo_2048",
    **kwargs,
) -> None:
    """Train a PPO agent on 2048."""
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise ImportError("Install RL dependencies: pip install game2048[rl]")
    
    from ..env import Game2048Env
    
    env = Monitor(Game2048Env())
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        **kwargs,
    )
    
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")