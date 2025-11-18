"""
Q-Learning Grid World Environment
A simple grid world where the agent learns to reach a goal while avoiding obstacles.
"""
import numpy as np
from typing import Tuple, List, Dict


class GridWorld:
    """Grid world environment for Q-learning."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Define special cells
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        # Set up grid values
        for obs in self.obstacles:
            if 0 <= obs[0] < size and 0 <= obs[1] < size:
                self.grid[obs] = -1  # Obstacle
        self.grid[self.goal] = 1  # Goal
        
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
        self.current_pos = self.start
        return self.current_pos
    
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
