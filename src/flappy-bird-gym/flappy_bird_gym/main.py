from cli import random_agent_env, rule_agent_env, qlearning_agent_env
import sys
import os

# Add the parent directory to Python path to find agents module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.rule_agent import RuleAgent
from agents.empty_qlearning_agent import YourQLearningAgent

if __name__ == '__main__':
    rule_agent = RuleAgent()
    q_agent = YourQLearningAgent()
    # rule_agent_env(rule_agent)      # Choose agent you want here (just comment out the line you don't want)
    qlearning_agent_env(q_agent)    