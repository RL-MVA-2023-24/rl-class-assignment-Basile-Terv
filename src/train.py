from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# from tqdm import trange
import numpy as np

device = torch.device("cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# TimeLimit object wraps an agent and overwrites its original step function to
# set trunc=True when max_episode_steps is reached (Ctrl+clic on TimeLimit)


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    # Below is an A2C agent 
    def __init__(self, config,policy_network, value_network):
        self.device = "cpu"
        self.policy = policy_network#policyNetwork(env)
        self.value = value_network#valueNetwork(env)
        self.scalar_dtype = next(policy_network.parameters()).dtype#next(self.policy.parameters()).dtype
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
        self.entropy_coefficient = config['entropy_coefficient'] if 'entropy_coefficient' in config.keys() else 0.001
    
    def sample_action(self, x):
        probabilities = self.policy(torch.as_tensor(x))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action.item(), log_prob, entropy
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        values = []
        entropies = []
        for ep in range(self.nb_episodes):
            print(f'episode {ep}/{self.nb_episodes}')
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                # print('len(rewards):',len(rewards))
                a, log_prob, entropy = self.sample_action(x)
                print('action:',a)
                y,r,d,trunc,info = env.step(a)
                values.append(self.value(torch.as_tensor(x)).squeeze(dim=0))
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                print('episode_cum_reward:',episode_cum_reward)
                if d or trunc:#might need to change this condition to handle trunc=True
                    # compute returns-to-go
                    new_returns = []
                    G_t = self.value(torch.as_tensor(x)).squeeze(dim=0)
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        advantages = returns - values
        pg_loss = -(advantages.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        critic_loss = advantages.pow(2).mean()
        loss = pg_loss + critic_loss + self.entropy_coefficient * entropy_loss
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in range(nb_rollouts):
            print('epoch:',ep,'\n')
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards

    def act(self, observation, use_random=False):
        probabilities = self.policy(torch.as_tensor(observation))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        return action.item()

    # To understand torch.save and torch.load:
    # https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE
    # To load the state_dict, we need an instantiation of the model. Self.policy and self.value are instantiated
    # with the ProjectAgent() class

    def save(self, path=''):
        torch.save(self.policy.state_dict(),'policy_network.pth')
        torch.save(self.value.state_dict(),'value_network.pth')

    def load(self):
        self.policy.load_state_dict(torch.load('policy_network.pth'))
        self.value.load_state_dict(torch.load('value_network.pth'))

"""
config = {'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 10,
          'entropy_coefficient': 1e-3
         }

pi = policyNetwork(env)
agent = reinforce_agent(config, pi)
returns = agent.train(env,50)
plt.plot(returns)
"""

class valueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.fc1 = nn.Linear(state_dim, 128).double()
        self.fc2 = nn.Linear(128, 1).double()

    def forward(self, x):
        # x=torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class policyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.fc1 = nn.Linear(state_dim, 128).double()
        self.fc2 = nn.Linear(128, n_action).double()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        # print('x type:',x.type())
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)

# https://realpython.com/if-name-main-python/
if __name__=='__main__':
    import matplotlib.pyplot as plt
    config = {'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 10,
          'entropy_coefficient': 1e-3
         }
    pi = policyNetwork(env)
    V = valueNetwork(env)
    agent = ProjectAgent(config, pi,V)
    returns = agent.train(env,2)
    plt.plot(returns)
    plt.savefig('train_plot')
    agent.save()