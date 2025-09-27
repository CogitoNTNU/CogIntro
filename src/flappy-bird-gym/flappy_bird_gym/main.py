from cli import random_agent_env, rule_agent_env
from agents.rule_agent import RuleAgent

if __name__ == '__main__':
    print('Vi er her!')
    rule_agent = RuleAgent()
    rule_agent_env(rule_agent)