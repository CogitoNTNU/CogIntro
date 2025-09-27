# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Handles the initialization of the game through the command line interface.
"""

import argparse
import time
import sys
import os

# Add the parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flappy_bird_gym
from agents.agent_factory import AgentFactory


def _get_args():
    """ Parses the command line arguments and returns them. """
    parser = argparse.ArgumentParser(description=__doc__)

    # Get available agent modes
    available_agents = AgentFactory.get_available_agents()
    all_modes = ["human", "random"] + available_agents

    # Argument for the mode of execution:
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="human",
        choices=all_modes,
        help="The execution mode for the game.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model file (for AI agents)",
    )

    return parser.parse_args()


def random_agent_env():
    env = flappy_bird_gym.make("FlappyBird-v0")
    obs, info = env.reset()
    score = 0
    while True:
        env.render()

        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, reward, terminated, truncated, info = env.step(action)

        score += reward
        print(f"Obs: {obs}\n"
              f"Action: {action}\n"
              f"Score: {score}\n")

        time.sleep(1 / 30)

        if terminated or truncated:
            env.render()
            time.sleep(0.5)
            break


def ai_agent_env(agent_type: str, model_path: str = None):
    """Run an AI agent in the environment."""
    try:
        # Create agent
        agent = AgentFactory.create_agent(agent_type)

        # Load model if provided
        if model_path:
            agent.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Running untrained {agent_type} agent (will use random actions initially)")

        # Run agent
        env = flappy_bird_gym.make("FlappyBird-v0")
        obs, info = env.reset()
        score = 0
        step_count = 0

        while True:
            env.render()

            # Get action from agent
            action = agent.select_action(obs)

            # Process step
            obs, reward, terminated, truncated, info = env.step(action)

            score += reward
            step_count += 1

            print(f"Step: {step_count}, Obs: {obs}, Action: {action}, Score: {score}")

            time.sleep(1 / 30)  # Control FPS

            if terminated or truncated:
                env.render()
                time.sleep(0.5)
                print(f"Game Over! Final Score: {score}")
                break

        env.close()

    except ImportError as e:
        print(f"Failed to import agent dependencies: {e}")
        print("Make sure PyTorch is installed: pip install torch")
    except Exception as e:
        print(f"Error running agent: {e}")


def main():
    args = _get_args()

    if args.mode == "human":
        flappy_bird_gym.original_game.main()
    elif args.mode == "random":
        random_agent_env()
    elif args.mode in AgentFactory.get_available_agents():
        ai_agent_env(args.mode, args.model_path)
    else:
        print("Invalid mode!")
