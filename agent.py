"""
Q-Learning Agent Implementation
"""
import numpy as np
from typing import Tuple, Dict, List
import random


class QLearningAgent:
    """Q-Learning agent for navigating the grid world."""
    
    def __init__(self, grid_size: int = 5, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.actions = ['up', 'down', 'left', 'right']
        self.q_table = np.zeros((grid_size, grid_size, len(self.actions)))
        
        # Training history
        self.episode_rewards = []
        self.episode_steps = []
    
    def get_action(self, state: Tuple[int, int]) -> str:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(self.actions)
        else:
            # Exploit: best action from Q-table
            action_index = np.argmax(self.q_table[state[0], state[1]])
            return self.actions[action_index]
    
    def update_q_value(self, state: Tuple[int, int], action: str, 
                       reward: float, next_state: Tuple[int, int]):
        """Update Q-value using Q-learning formula."""
        action_index = self.actions.index(action)
        
        # Q-learning update rule
        current_q = self.q_table[state[0], state[1], action_index]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state[0], state[1], action_index] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_table_for_visualization(self) -> List[List[List[float]]]:
        """Convert Q-table to list format for JSON serialization."""
        return self.q_table.tolist()
    
    def get_policy(self) -> List[List[str]]:
        """Get the current policy (best action for each state)."""
        policy = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                best_action_index = np.argmax(self.q_table[i, j])
                row.append(self.actions[best_action_index])
            policy.append(row)
        return policy
    
    def get_state_values(self) -> List[List[float]]:
        """Get the maximum Q-value for each state."""
        values = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                row.append(float(np.max(self.q_table[i, j])))
            values.append(row)
        return values
