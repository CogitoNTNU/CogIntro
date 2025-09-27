"""
Agent Factory for creating different types of agents
"""

from typing import Dict, Type
from .base_agent import BaseAgent


class AgentFactory:
    """Factory class for creating agents."""

    # Registry of available agents
    _agents: Dict[str, Type[BaseAgent]] = {
        # Add agents here as they are implemented
        # 'ppo': PPOAgent,
        # 'rule-based': RuleBasedAgent,
        # 'genetic': GeneticAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: str, **kwargs) -> BaseAgent:
        """
        Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create
            **kwargs: Additional arguments to pass to the agent constructor

        Returns:
            Initialized agent instance

        Raises:
            ValueError: If agent_type is not supported
        """
        if agent_type not in cls._agents:
            available = list(cls._agents.keys())
            raise ValueError(f"Agent type '{agent_type}' not supported. Available: {available}")

        agent_class = cls._agents[agent_type]
        return agent_class(**kwargs)

    @classmethod
    def get_available_agents(cls) -> list:
        """Get list of available agent types."""
        return list(cls._agents.keys())

    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]):
        """
        Register a new agent type.

        Args:
            name: Name for the agent type
            agent_class: Agent class to register
        """
        cls._agents[name] = agent_class