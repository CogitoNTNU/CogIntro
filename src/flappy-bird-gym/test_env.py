#!/usr/bin/env python3

import flappy_bird_gym


def test_simple_env():
    """Test the simple numerical environment."""
    print("Testing FlappyBird-v0 (simple observations)...")
    env = flappy_bird_gym.make("FlappyBird-v0")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i + 1}: action={action}, obs={obs}, reward={reward}, terminated={terminated}"
        )

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            break

    env.close()
    print("Simple environment test completed!\n")


def test_rgb_env():
    """Test the RGB image environment."""
    print("Testing FlappyBird-rgb-v0 (RGB observations)...")
    env = flappy_bird_gym.make("FlappyBird-rgb-v0")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i + 1}: action={action}, obs_shape={obs.shape}, reward={reward}, terminated={terminated}"
        )

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            break

    env.close()
    print("RGB environment test completed!")


if __name__ == "__main__":
    test_simple_env()
    test_rgb_env()
    print("\nAll tests completed successfully!")
