"""
Simple DQN Agent for Flappy Bird
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from typing import Dict, Any
import flappy_bird_gym
from .base_agent import BaseAgent


class DQNetwork(nn.Module):
    """Simple neural network for DQN."""

    def __init__(self, input_size: int = 2, hidden_size: int = 64, output_size: int = 2):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for Flappy Bird."""

    def __init__(self,
                 env_name: str = "FlappyBird-v0",
                 learning_rate: float = 0.001,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 memory_size: int = 10000):
        """
        Initialize DQN agent.

        Args:
            env_name: Environment name
            learning_rate: Learning rate for optimizer
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            gamma: Discount factor
            batch_size: Training batch size
            memory_size: Replay buffer size
        """
        super().__init__(env_name)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork().to(self.device)
        self.target_network = DQNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = ReplayBuffer(memory_size)

        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'losses': [],
            'scores': [],
            'epsilons': []
        }

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, observation: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.choice(2)  # Random action

        # Convert observation to tensor
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state)

        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Store loss
        self.training_stats['losses'].append(loss.item())
        self.training_stats['epsilons'].append(self.epsilon)

    def train(self, episodes: int = 1000, target_update_freq: int = 100) -> Dict[str, Any]:
        """Train the DQN agent."""
        env = flappy_bird_gym.make(self.env_name)

        print(f"Training DQN agent for {episodes} episodes...")
        print(f"Device: {self.device}")

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0

            while True:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Modify reward to help learning
                modified_reward = reward
                if terminated:  # Bird died
                    modified_reward = -100

                # Store experience
                self.remember(state, action, modified_reward, next_state, terminated or truncated)

                # Train
                self.replay()

                state = next_state
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Update target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()

            # Store statistics
            self.training_stats['scores'].append(total_reward)
            self.training_stats['episodes'] = episode + 1
            self.training_stats['total_steps'] += steps

            # Print progress
            if episode % 100 == 0:
                avg_score = np.mean(self.training_stats['scores'][-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}")

        env.close()

        # Final statistics
        final_avg = np.mean(self.training_stats['scores'][-100:])
        print(f"Training completed! Final average score (last 100 episodes): {final_avg:.2f}")

        return self.training_stats

    def save(self, filepath: str) -> None:
        """Save the agent's parameters."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load the agent's parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {filepath}")