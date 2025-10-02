"""
Base Agent class for Flappy Bird environment
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """Abstract base class for all Flappy Bird agents."""

    def __init__(self, env_name: str = "FlappyBird-v0"):
        """
        Initialize the base agent.

        Args:
            env_name: Name of the environment to use
        """
        self.env_name = env_name
        self.env = None

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.

        Args:
            observation: Current environment observation

        Returns:
            Action to take (0: do nothing, 1: flap)
        """
        pass

    @abstractmethod
    def train(self, episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the agent for a specified number of episodes.

        Args:
            episodes: Number of training episodes

        Returns:
            Training statistics dictionary
        """
        pass

    def evaluate(self, episodes: int = 100, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent's performance.

        Args:
            episodes: Number of evaluation episodes
            render: Whether to render the environment

        Returns:
            Evaluation statistics
        """
        import flappy_bird_gym

        if self.env is None:
            self.env = flappy_bird_gym.make(self.env_name)

        scores = []
        episode_lengths = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            steps = 0

            while True:
                if render:
                    self.env.render()

                action = self.select_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            scores.append(total_reward)
            episode_lengths.append(steps)

        self.env.close()

        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "mean_episode_length": np.mean(episode_lengths),
            "episodes_evaluated": episodes,
        }

    def save(self, filepath: str) -> None:
        """Save the agent's parameters."""
        pass

    def load(self, filepath: str) -> None:
        """Load the agent's parameters."""
        pass
