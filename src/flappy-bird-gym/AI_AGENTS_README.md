# AI Agents for Flappy Bird Gym

This project now includes a modular AI agent system that makes it easy to train and test different reinforcement learning models on the Flappy Bird environment.

## Quick Start

### Install Dependencies
```bash
pip install torch  # or torch-cpu for smaller installation
```

### Train a DQN Agent
```bash
python train_agent.py --mode dqn --episodes 1000
```

### Evaluate a Trained Agent
```bash
python evaluate_agent.py --model-path models/dqn_agent.pth --episodes 10
```

### Run a Trained Agent in the GUI
```bash
flappy_bird_gym --mode dqn --model-path models/dqn_agent.pth
```

## Available Commands

### Training
```bash
python train_agent.py [OPTIONS]
```

Options:
- `--mode`: Agent type to train (`dqn`)
- `--episodes`: Number of training episodes (default: 1000)
- `--save-path`: Directory to save models (default: `models/`)
- `--eval-episodes`: Episodes for post-training evaluation (default: 100)

### Evaluation
```bash
python evaluate_agent.py [OPTIONS]
```

Options:
- `--mode`: Agent type (`dqn`)
- `--model-path`: Path to trained model (required)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Show the game while evaluating

### Playing with CLI
```bash
flappy_bird_gym --mode [MODE] [OPTIONS]
```

Available modes:
- `human`: Play manually
- `random`: Random agent
- `dqn`: DQN agent (requires `--model-path` for trained models)

## Project Structure

```
agents/
├── __init__.py              # Agent registry
├── base_agent.py           # Abstract base class for all agents
├── dqn_agent.py            # Deep Q-Network implementation
└── agent_factory.py        # Factory for creating agents
```

## Adding New Agents

To add a new agent type (e.g., PPO, A3C, genetic algorithm):

1. **Create the agent class** in `agents/your_agent.py`:
```python
from .base_agent import BaseAgent

class YourAgent(BaseAgent):
    def select_action(self, observation):
        # Your action selection logic
        pass

    def train(self, episodes):
        # Your training logic
        pass
```

2. **Register the agent** in `agents/__init__.py`:
```python
from .your_agent import YourAgent
__all__.append('YourAgent')
```

3. **Add to factory** in `agents/agent_factory.py`:
```python
from .your_agent import YourAgent

class AgentFactory:
    _agents = {
        'dqn': DQNAgent,
        'your-agent': YourAgent,  # Add this line
        # ...
    }
```

4. **Use immediately**:
```bash
python train_agent.py --mode your-agent
flappy_bird_gym --mode your-agent --model-path models/your_agent.pth
```

## DQN Agent Details

The included DQN (Deep Q-Network) agent:
- Uses a simple 3-layer neural network
- Implements experience replay with a buffer of 10,000 experiences
- Uses epsilon-greedy exploration with decay
- Targets network updates every 100 episodes
- Reward modification: -100 for dying, +1 for each step

### Hyperparameters
- Learning rate: 0.001
- Epsilon decay: 0.995 (min 0.01)
- Discount factor (gamma): 0.99
- Batch size: 32
- Hidden layer size: 64

## Performance

With the default settings, the DQN agent typically achieves:
- Random agent: ~5-20 score
- DQN after 1000 episodes: ~50-150 score
- DQN after 5000 episodes: Often >200 score

## Environment Details

The `FlappyBird-v0` environment provides simple observations:
- `observation[0]`: Horizontal distance to next pipe (normalized)
- `observation[1]`: Vertical distance to next pipe gap (normalized)

Actions:
- `0`: Do nothing
- `1`: Flap