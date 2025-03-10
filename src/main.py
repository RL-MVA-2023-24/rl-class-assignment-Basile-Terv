import random
import os
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent#, policyNetwork, valueNetwork  # Replace DummyAgent with your agent implementation
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    """env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
    config = {'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 10,
          'entropy_coefficient': 1e-3
         }
    pi = policyNetwork(env)
    V = valueNetwork(env)"""
    agent = ProjectAgent()
    # il aurait été équivalent d'instancier policyNetwork et valueNetwork comme
    # agent=ProjectAgent(policy_network=policyNetwork(),value_network=valueNetwork(),config={..})
    # agent = ProjectAgent()
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
