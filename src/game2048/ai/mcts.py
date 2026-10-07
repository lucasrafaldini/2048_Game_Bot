"""Monte Carlo Tree Search (MCTS) agent for 2048."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .base import BaseAgent
from ..core.board import Board2048, simulate_move
from ..core.constants import Direction, GameState


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 100
    time_limit: float = 0.1  # seconds
    exploration_constant: float = 1.414  # sqrt(2)
    max_depth: int = 50
    use_heuristic_rollout: bool = True
    progressive_widening: bool = True
    pw_alpha: float = 0.5
    pw_k: float = 10.0


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    board: Board2048
    parent: Optional["MCTSNode"] = None
    move: Optional[Direction] = None
    is_chance: bool = False  # True for chance nodes (tile spawns)
    
    visits: int = 0
    total_value: float = 0.0
    children: dict = field(default_factory=dict)  # move/spawn -> node
    untried_moves: list[Direction] = field(default_factory=list)
    untried_spawns: list[tuple[int, int, int]] = field(default_factory=list)  # (r, c, value)
    
    def __post_init__(self):
        if self.is_chance:
            self._init_chance()
        else:
            self._init_player()
    
    def _init_player(self):
        self.untried_moves = self.board.get_valid_moves()
    
    def _init_chance(self):
        empty = self.board.empty_cells
        for r, c in empty:
            self.untried_spawns.append((r, c, 2))
            self.untried_spawns.append((r, c, 4))
    
    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0
    
    def ucb1(self, exploration: float) -> float:
        """UCB1 formula for node selection."""
        if self.visits == 0:
            return float('inf')
        return self.value + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def is_fully_expanded(self) -> bool:
        if self.is_chance:
            return len(self.untried_spawns) == 0
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration: float) -> "MCTSNode":
        """Select best child using UCB1."""
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search agent for 2048.
    
    Alternates between player nodes (move selection) and chance nodes (tile spawns).
    Uses UCB1 for exploration/exploitation balance.
    """
    
    def __init__(self, config: MCTSConfig | None = None, name: str | None = None):
        super().__init__(name or "MCTS")
        self.config = config or MCTSConfig()
        self._rng = random.Random()
        self._heuristic_cache: dict[int, float] = {}
    
    def get_move(self, board: Board2048) -> Direction:
        self._nodes_evaluated = 0
        self._max_depth_reached = 0
        self._heuristic_cache.clear()
        
        root = MCTSNode(board.copy())
        start_time = time.perf_counter()
        
        simulations = 0
        while True:
            if self.config.time_limit > 0:
                if time.perf_counter() - start_time > self.config.time_limit:
                    break
            elif simulations >= self.config.num_simulations:
                break
            
            self._simulate(root)
            simulations += 1
        
        # Select best move by visit count (most robust)
        if not root.children:
            valid = board.get_valid_moves()
            return valid[0] if valid else Direction.UP
        
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visits)
        self._nodes_evaluated = simulations
        return best_move
    
    def _simulate(self, root: MCTSNode) -> None:
        """Run one MCTS simulation."""
        node = root
        depth = 0
        
        # Selection
        while node.is_fully_expanded() and node.children and depth < self.config.max_depth:
            if node.is_chance:
                # For chance nodes, sample by probability
                node = self._select_chance_child(node)
            else:
                node = node.best_child(self.config.exploration_constant)
            depth += 1
        
        # Expansion
        if depth < self.config.max_depth and node.children.get("terminal") is None:
            if node.is_chance:
                node = self._expand_chance(node)
            else:
                node = self._expand_player(node)
            depth += 1
        
        # Simulation (Rollout)
        if node.children.get("terminal") is None:
            value = self._rollout(node.board, node.is_chance, depth)
        else:
            value = node.children["terminal"].value
        
        # Backpropagation
        self._backpropagate(node, value)
    
    def _select_chance_child(self, node: MCTSNode) -> MCTSNode:
        """Select chance child by probability (2: 0.9, 4: 0.1)."""
        # Weighted random selection
        total_weight = 0.0
        for child in node.children.values():
            if child.move is not None:  # spawn node
                weight = 0.9 if child.move.value == 2 else 0.1  # hack: store value in move
                total_weight += weight
        
        r = self._rng.random() * total_weight
        cumulative = 0.0
        for child in node.children.values():
            if child.move is not None:
                weight = 0.9 if child.move.value == 2 else 0.1
                cumulative += weight
                if r <= cumulative:
                    return child
        return next(iter(node.children.values()))
    
    def _expand_player(self, node: MCTSNode) -> MCTSNode:
        """Expand player node with a new move."""
        if not node.untried_moves:
            return node
        
        move = self._rng.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        new_board, _ = simulate_move(node.board, move)
        state = new_board.get_state()
        
        if state != "playing":
            # Terminal node
            terminal = MCTSNode(new_board, node, move)
            terminal.visits = 1
            terminal.total_value = self._terminal_value(new_board, state)
            node.children[move] = terminal
            node.children["terminal"] = terminal
            return terminal
        
        # Create chance node for next tile spawn
        chance_node = MCTSNode(new_board, node, move, is_chance=True)
        node.children[move] = chance_node
        return chance_node
    
    def _expand_chance(self, node: MCTSNode) -> MCTSNode:
        """Expand chance node with a new tile spawn."""
        if not node.untried_spawns:
            return node
        
        spawn = self._rng.choice(node.untried_spawns)
        node.untried_spawns.remove(spawn)
        r, c, value = spawn
        
        new_board = node.board.copy()
        new_board._grid[r][c] = value
        state = new_board.get_state()
        
        if state != "playing":
            terminal = MCTSNode(new_board, node, None)  # terminal node
            terminal.visits = 1
            terminal.total_value = self._terminal_value(new_board, state)
            node.children[spawn] = terminal
            node.children["terminal"] = terminal
            return terminal
        
        # Create player node (move=None for chance nodes)
        player_node = MCTSNode(new_board, node, None, is_chance=False)
        node.children[spawn] = player_node
        return player_node
    
    def _rollout(self, board: Board2048, is_chance: bool, depth: int) -> float:
        """Random rollout to terminal state or max depth."""
        current = board.copy()
        current_is_chance = is_chance
        
        for _ in range(self.config.max_depth - depth):
            state = current.get_state()
            if state != "playing":
                return self._terminal_value(current, state)
            
            if current_is_chance:
                # Random tile spawn
                empty = current.empty_cells
                if not empty:
                    return self._evaluate(current)
                r, c = self._rng.choice(empty)
                value = self._rng.choices([2, 4], weights=[0.9, 0.1], k=1)[0]
                current._grid[r][c] = value
            else:
                # Random move
                moves = current.get_valid_moves()
                if not moves:
                    return self._evaluate(current)
                move = self._rng.choice(moves)
                current, _ = simulate_move(current, move)
            
            current_is_chance = not current_is_chance
        
        return self._evaluate(current)
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
    
    def _evaluate(self, board: Board2048) -> float:
        """Fast heuristic evaluation with caching."""
        board_hash = hash(board)
        if board_hash in self._heuristic_cache:
            return self._heuristic_cache[board_hash]
        
        # Simple but fast heuristic
        grid = board.grid
        score = 0.0
        
        # Empty tiles
        empty = len(board.empty_cells)
        score += empty * 10
        
        # Max tile
        score += board.max_tile
        
        # Monotonicity (simplified)
        for r in range(4):
            row = grid[r]
            for c in range(3):
                if row[c] >= row[c + 1] and row[c] > 0:
                    score += row[c] * 0.1
        
        for c in range(4):
            for r in range(3):
                if grid[r][c] >= grid[r + 1][c] and grid[r][c] > 0:
                    score += grid[r][c] * 0.1
        
        # Corner bonus
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        max_tile = board.max_tile
        for r, c in corners:
            if grid[r][c] == max_tile:
                score += 100
        
        self._heuristic_cache[board_hash] = score
        return score
    
    def _terminal_value(self, board: Board2048, state: GameState) -> float:
        if state == "won":
            return 100000 + board.score
        elif state == "lost":
            return -100000
        return self._evaluate(board)


def create_mcts_agent(
    simulations: int = 100,
    time_limit: float = 0.1,
    exploration: float = 1.414,
) -> MCTSAgent:
    """Create MCTS agent with common settings."""
    config = MCTSConfig(
        num_simulations=simulations,
        time_limit=time_limit,
        exploration_constant=exploration,
    )
    return MCTSAgent(config)