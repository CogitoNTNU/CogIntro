#!/usr/bin/env python3
"""
Training script for Flappy Bird agents
"""

import argparse
import os
from agents.agent_factory import AgentFactory


def main():
    parser = argparse.ArgumentParser(description="Train a Flappy Bird agent")
    parser.add_argument("--mode", "-m",
                       choices=AgentFactory.get_available_agents(),
                       default="dqn",
                       help="Agent type to train")
    parser.add_argument("--episodes", "-e",
                       type=int,
                       default=1000,
                       help="Number of training episodes")
    parser.add_argument("--save-path", "-s",
                       type=str,
                       default="models/",
                       help="Directory to save trained models")
    parser.add_argument("--eval-episodes",
                       type=int,
                       default=100,
                       help="Number of evaluation episodes after training")

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Create agent
    print(f"Creating {args.mode} agent...")
    agent = AgentFactory.create_agent(args.mode)

    # Train agent
    print(f"Training for {args.episodes} episodes...")
    training_stats = agent.train(episodes=args.episodes)

    # Save model
    model_path = os.path.join(args.save_path, f"{args.mode}_agent.pth")
    agent.save(model_path)

    # Evaluate agent
    print(f"Evaluating agent over {args.eval_episodes} episodes...")
    eval_stats = agent.evaluate(episodes=args.eval_episodes)

    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Agent Type: {args.mode}")
    print(f"Training Episodes: {training_stats['episodes']}")
    print(f"Total Training Steps: {training_stats['total_steps']}")
    print(f"Final Average Score (last 100): {sum(training_stats['scores'][-100:])/len(training_stats['scores'][-100:]):.2f}")

    print(f"\nEvaluation Results:")
    print(f"Mean Score: {eval_stats['mean_score']:.2f} ± {eval_stats['std_score']:.2f}")
    print(f"Max Score: {eval_stats['max_score']:.0f}")
    print(f"Min Score: {eval_stats['min_score']:.0f}")
    print(f"Mean Episode Length: {eval_stats['mean_episode_length']:.1f}")

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()