"""Gymnasium environment for 2048 RL training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core.board import Board2048, simulate_move
from .core.constants import Direction, GameState
from .core.game import Game2048


class Game2048Env(gym.Env):
    """
    Gymnasium environment for 2048.
    
    Observation: 4x4x16 one-hot encoding (log2 tile values)
    Action: Discrete(4) - UP, RIGHT, DOWN, LEFT
    Reward: Score gained + heuristic shaping
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}
    
    def __init__(
        self,
        render_mode: str | None = None,
        reward_type: str = "score",  # "score", "log_score", "shaped"
        max_steps: int = 1000,
        seed: int | None = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.max_steps = max_steps
        
        # Observation: 4x4x16 one-hot (log2 values 1-15, 0=empty)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, 4, 16), dtype=np.float32
        )
        
        # Action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)
        
        self._game = Game2048(seed=seed)
        self._step_count = 0
        self._prev_score = 0
        self._prev_max_tile = 0
        
        # Direction mapping
        self._actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._game = Game2048(seed=seed)
        else:
            self._game.reset()
        
        self._step_count = 0
        self._prev_score = 0
        self._prev_max_tile = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, info
    
    def step(self, action: int):
        direction = self._actions[action]
        
        # Execute move
        changed, state = self._game.step(direction)
        self._step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(changed, state)
        
        # Check termination
        terminated = state in ("won", "lost")
        truncated = self._step_count >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, changed: bool, state: GameState) -> float:
        """Calculate reward based on reward_type."""
        if self.reward_type == "score":
            # Raw score difference
            reward = self._game.board.score - self._prev_score
            self._prev_score = self._game.board.score
            return float(reward)
        
        elif self.reward_type == "log_score":
            # Log of score gained
            gained = self._game.board.score - self._prev_score
            self._prev_score = self._game.board.score
            return float(np.log1p(gained)) if gained > 0 else 0.0
        
        elif self.reward_type == "shaped":
            # Shaped reward with heuristics
            reward = 0.0
            
            # Score gained
            gained = self._game.board.score - self._prev_score
            self._prev_score = self._game.board.score
            reward += gained * 0.01
            
            # Max tile bonus
            if self._game.board.max_tile > self._prev_max_tile:
                reward += np.log2(self._game.board.max_tile) * 10
                self._prev_max_tile = self._game.board.max_tile
            
            # Empty tile bonus
            reward += len(self._game.board.empty_cells) * 0.1
            
            # Invalid move penalty
            if not changed:
                reward -= 1.0
            
            # Terminal rewards
            if state == "won":
                reward += 1000.0
            elif state == "lost":
                reward -= 100.0
            
            return reward
        
        return 0.0
    
    def _get_obs(self) -> np.ndarray:
        """Get observation as 4x4x16 one-hot."""
        grid = self._game.board.to_numpy()
        obs = np.zeros((4, 4, 16), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                val = grid[r, c]
                if val > 0:
                    idx = int(np.log2(val))
                    if idx < 16:
                        obs[r, c, idx] = 1.0
        return obs
    
    def _get_info(self) -> dict:
        return {
            "score": self._game.board.score,
            "max_tile": self._game.board.max_tile,
            "moves": self._game.board.moves,
            "state": self._game.board.get_state(),
            "step": self._step_count,
        }
    
    def render(self):
        if self.render_mode == "ansi":
            return str(self._game.board)
        elif self.render_mode == "human":
            print(self._game.board)
            print(f"Score: {self._game.board.score} | Max: {self._game.board.max_tile} | Moves: {self._game.board.moves}")
    
    def close(self):
        pass


# Register environment
gym.register(
    id="Game2048-v0",
    entry_point="game2048.env:Game2048Env",
    max_episode_steps=1000,
)