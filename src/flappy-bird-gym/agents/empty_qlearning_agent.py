from typing import Any, Dict
import numpy as np
from agents.base_agent import BaseAgent
import flappy_bird_gym
import time


class YourQLearningAgent(BaseAgent):
    def __init__(
        self,
        # TODO: Define how big your Q-table will be:
        # OBS: Lager table needs longer training
        n_x_states=50,  # How many x-distances your table will have
        n_y_states=100,  # How many y-distances your table will have
        n_actions=2,  # There are only two actions - 0: nothing, 1: flap
        env_name: str = "FlappyBird-v0",
        # TODO: You can play around with these values too :)
        learning_rate=0.2,  # How much the table changes after every action
        discount_factor=1,  # The importance of future reward
        exploration_prob=0.2,  # How often the agent does a random move
        epochs=1000,  # How many times we play the game during training
    ):
        self.env_name = env_name
        self.env = flappy_bird_gym.make(env_name)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epochs = epochs

        # Model parameters
        self.n_x_states = n_x_states
        self.n_y_states = n_y_states
        self.n_actions = n_actions

        # Q-table
        self.Q_table = np.zeros((n_x_states, n_y_states, n_actions))

    def process_observation_values(self, observation: np.ndarray):
        # TODO: Map x and y values to indices in the Q-table
        # x distance (observed by Maia, might be wrong): [0, 1.7]
        # y distance (observed by Maia, might be wrong): [-0.8, 0.8]

        pass

    def select_action(self, observation: np.ndarray) -> int:
        # TODO: Choose an action
        # 0: Ingenting
        # 1: flap
        # Hint: Here we want to use the Q-table. In our current state, we want to choose the action with the largest Q-value

        pass

    def train(self, episodes: int = 1000) -> Dict[str, Any]:
        best_score = 0
        best_episode = 0

        # Epochs
        for episode in range(episodes):
            print(f"Episode: {episode}")

            # TODO: Update learning rate and exploration rate, if you want :)
            # Here you can try different strategies!
            # For example you can start with learning rate = max(0.05, learing rate * 0.999)
            # And exploration = max(0.005, exploration * 0.999)
            # Or just nothing
            # You decide :)

            self.learning_rate = max(0.05, self.learning_rate * 0.999)
            self.exploration_prob = max(0.05, self.exploration_prob * 0.999)

            env = self.env
            obs, info = env.reset()
            score = 0

            # While the bird lives
            while True:
                env.render()

                # Process state values (get Q-table indices)
                x_idx, y_idx = self.process_observation_values(obs)

                # Getting action based on Q-table values:
                # TODO: Use the exploration probability to either choose a random action, or an action based on the Q-table
                action = 0

                # Processing the action:
                obs, reward, terminated, truncated, info = env.step(action)

                # Rewarding our bird
                # Current rewards:
                # One point if it lives
                # -100 points if it dies

                # TODO: Feel free to play with these values :)
                if terminated:
                    reward = -100
                else:
                    reward = 1

                # Process new state values (get Q-table indices)
                new_x_idx, new_y_idx = self.process_observation_values(obs)

                # TODO: Time to learn! Complete the TODO's below
                # Here we will update the Q-table
                # Maia was very confused for a long time, so no worries if this part is hard
                # We love questions :)

                if terminated:
                    # No future rewards if terminal state
                    pass
                else:
                    # TODO: Something here ;) Hint: The formula from the presentation is a nice place to start
                    pass

                # TODO: Use our beloved formula, and update the Q-table!
                self.Q_table[x_idx, y_idx, action] = 0

                score += reward

                time.sleep(1 / 30)

                if terminated or truncated:
                    env.render()
                    time.sleep(0.5)
                    break

            if score > best_score:
                best_score = score
                best_episode = episode

            print(f"Obs: {obs}\nScore: {score}\n")

        # Print best performing episode, and the score
        print(f"Best score: {best_score}")
        print(f"Acheived in episode: {best_episode}")

        return
