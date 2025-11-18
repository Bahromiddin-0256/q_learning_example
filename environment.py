"""
Q-Learning Grid World Environment
A simple grid world where the agent learns to reach a goal while avoiding obstacles.
"""
import numpy as np
from typing import Tuple, List, Dict


class GridWorld:
    """Grid world environment for Q-learning."""

    def __init__(
        self,
        size: int = 5,
        obstacle_count: int = 3,
        obstacle_positions: List[Tuple[int, int]] = None,
        seed: int = None,
        randomize_on_reset: bool = False,
    ):
        self.size = size
        self.grid = np.zeros((size, size))

        # Define special cells
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # RNG for obstacle placement (kept so placements can be reproducible)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.randomize_on_reset = randomize_on_reset

        # Obstacles: either provided or randomly placed
        if obstacle_positions is not None:
            # validate and clamp to grid
            valid_obs = []
            for obs in obstacle_positions:
                if 0 <= obs[0] < size and 0 <= obs[1] < size:
                    if obs != self.start and obs != self.goal:
                        valid_obs.append(tuple(obs))
            self.obstacles = valid_obs
        else:
            self.obstacles = []
            self._place_obstacles(obstacle_count)

        # Set up grid values
        self._refresh_grid()

        self.current_pos = self.start
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment to starting position."""
        # Optionally randomize obstacles each reset
        if self.randomize_on_reset:
            # place obstacles using the same RNG so seed is reproducible
            # note: _place_obstacles will clamp count to available cells
            self._place_obstacles(len(self.obstacles) or 3)
            self._refresh_grid()

        self.current_pos = self.start
        return self.current_pos

    def _place_obstacles(self, count: int = 3):
        """Place `count` obstacles randomly on the grid excluding start/goal."""
        max_available = self.size * self.size - 2
        count = min(count, max(0, max_available))

        # All possible positions excluding start and goal
        positions = [(r, c) for r in range(self.size) for c in range(self.size)
                     if (r, c) != self.start and (r, c) != self.goal]

        if count == 0:
            self.obstacles = []
            return

        # Choose without replacement
        indices = self._rng.choice(len(positions), size=count, replace=False)
        self.obstacles = [positions[i] for i in indices]

    def _refresh_grid(self):
        """Refresh the numeric grid representation based on start/goal/obstacles."""
        self.grid = np.zeros((self.size, self.size))
        for obs in self.obstacles:
            if 0 <= obs[0] < self.size and 0 <= obs[1] < self.size:
                self.grid[obs] = -1  # Obstacle
        self.grid[self.goal] = 1  # Goal
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment.
        Returns: (new_position, reward, done)
        """
        effect = self.action_effects[action]
        new_row = self.current_pos[0] + effect[0]
        new_col = self.current_pos[1] + effect[1]
        
        # Check if new position is valid
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # Hit wall - stay in place
            return self.current_pos, -0.1, False
        
        new_pos = (new_row, new_col)
        
        # Check if hit obstacle
        if new_pos in self.obstacles:
            return self.current_pos, -1.0, False
        
        # Update position
        self.current_pos = new_pos
        
        # Check if reached goal
        if self.current_pos == self.goal:
            return self.current_pos, 10.0, True
        
        # Normal step
        return self.current_pos, -0.04, False
    
    def get_state_representation(self) -> Dict:
        """Get current state as dictionary for visualization."""
        return {
            'grid_size': self.size,
            'current_pos': self.current_pos,
            'start': self.start,
            'goal': self.goal,
            'obstacles': self.obstacles
        }
