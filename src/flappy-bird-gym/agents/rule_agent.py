"""
Rule based Agent class for Flappy Bird environment
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
from agents.base_agent import BaseAgent
import flappy_bird_gym


class RuleAgent(BaseAgent):
    """Abstract base class for all Flappy Bird agents."""
    def __init__(self, env_name: str = "FlappyBird-v0"):
        """
        Initialize the base agent.

        Args:
            env_name: Name of the environment to use
        """
        self.env_name = env_name
        self.env = flappy_bird_gym.make(env_name)
        print(self.env)

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.

        Args:
            observation: Current environment observation

        Returns:
            Action to take (0: do nothing, 1: flap)
        """
        return 1 if observation[1]+0.04 < 0 else 0

    def train(self, episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.

        Args:
            episodes: Number of training episodes

        Returns:
            Training statistics dictionary
        """
        return {}
