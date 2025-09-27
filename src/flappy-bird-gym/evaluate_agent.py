#!/usr/bin/env python3
"""
Evaluation script for trained Flappy Bird agents
"""

import argparse
from agents.agent_factory import AgentFactory


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird agent")
    parser.add_argument("--mode", "-m",
                       choices=AgentFactory.get_available_agents(),
                       default="dqn",
                       help="Agent type to evaluate")
    parser.add_argument("--model-path", "-p",
                       type=str,
                       required=True,
                       help="Path to trained model file")
    parser.add_argument("--episodes", "-e",
                       type=int,
                       default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", "-r",
                       action="store_true",
                       help="Render the environment during evaluation")

    args = parser.parse_args()

    # Create agent
    print(f"Creating {args.mode} agent...")
    agent = AgentFactory.create_agent(args.mode)

    # Load model
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)

    # Evaluate
    print(f"Evaluating for {args.episodes} episodes...")
    results = agent.evaluate(episodes=args.episodes, render=args.render)

    # Print results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Episodes: {results['episodes_evaluated']}")
    print(f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"Max Score: {results['max_score']:.0f}")
    print(f"Min Score: {results['min_score']:.0f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f}")


if __name__ == "__main__":
    main()