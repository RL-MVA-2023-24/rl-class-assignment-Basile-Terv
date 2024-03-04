from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from copy import deepcopy
import numpy as np
import os
import time

# ------donné par Adil--------------------------------------
from functools import partial
# f(x,y) fonction prédéf. partial renvoie nouvelle fonction avec y fixé
# on peut le faire pour une classe, pour overwrite un init. Mieux que 
# MyClass_new = partial(MyClass, some_arg=1)
# -----------------------------------------------------------

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# TimeLimit object wraps an agent and overwrites its original step function to
# set trunc=True when max_episode_steps is reached (Ctrl+clic on TimeLimit)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
# ---------------------------------------------------------------------------
# Code inspired by RL6 class
class ProjectAgent_A2C:
    # Below is an A2C agent 
    def __init__(self, config,policy_network, value_network):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = policy_network.to(self.device) #policyNetwork(env)
        self.value = value_network.to(self.device) #valueNetwork(env)
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
                print("x:",x)
                a, log_prob, entropy = self.sample_action(x)
                print('action:',a)
                print("log_prob:",log_prob)
                print("entropy:",entropy)
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
                    # compute bootstrapped Monte-Carlo returns new_returns in reverse sense
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    # reverse to get new_returns in chronological order
                    new_returns = list(reversed(new_returns))
                    # add returns of current episode to returns list of current gradient step
                    returns.extend(new_returns)
                    # one gradient step
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
    
# --------------------------------------------------------------------
# Code inspired by RL4 class
from dqn_greedy_action import greedy_action
from replay_buffer2 import ReplayBuffer

class ProjectAgent_DQN:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring: takes a lot of time:
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return greedy_action(self.model, observation)
    
    def save(self, path=f"{os.getcwd()}/src/DQN_bigger_longer.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/src/DQN_bigger_longer.pth", map_location='cpu'))
        # turn off potential batch norm, dropout, etc. for inference time
        self.model.eval()

class DQN(nn.Module):
    def __init__(self, env, hidden_size=256, depth=6):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.input_layer = nn.Linear(state_dim, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(hidden_size, n_action)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)

# --------------------------------------------------------------------------------
# on définit une classe wrapper de la classe _ProjectAgent où on fixe
# des arguments
"""
ProjectAgent = partial(ProjectAgent_A2C,config={'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 10,
          'entropy_coefficient': 1e-3
         } ,policy_network=policyNetwork(env), value_network=valueNetwork(env))"""

# ---------------------------------------------------------
# Declare network
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dqn=DQN(env, hidden_size=512, depth=7)

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 100000,
          'epsilon_min': 0.02,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 100,
          'batch_size': 800,
          'gradient_steps': 5,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 0}

"""config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.98,
                'buffer_size': 100000,
                'epsilon_min': 0.02,
                'epsilon_max': 1.,
                'epsilon_decay_period': 20000, # go plus haut? plus bas ?
                'epsilon_delay_decay': 100,
                'batch_size': 800,
                'gradient_steps': 3,
                'update_target_strategy': 'replace', # or 'ema'
                'update_target_freq': 400,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss()}"""

ProjectAgent = partial(ProjectAgent_DQN,config=config,\
                       model=dqn)

# ------------------------------------------------------------------------
# !!!!!!!!!CAREFUL!!!!!!!!!!!!: always run this script from root of the repo
# !!!NOT!!!! from src

# https://realpython.com/if-name-main-python/
if __name__=='__main__':
    import matplotlib.pyplot as plt
    agent = ProjectAgent()#config, pi,V
    start_time = time.time()
    # for A2C: returns = agent.train(env,6)
    # for DQN:
    ep_return, disc_rewards, tot_rewards, V0 = agent.train(env,200)
    print("MC eval of total reward",tot_rewards)
    print("MC eval of discounted reward",disc_rewards)
    print("average $max_a Q(s_0)$",V0)
    end_time = time.time()  # Record end time
    training_time = end_time - start_time  # Calculate training time
    # Print training time
    print("Time taken for training: {:.2f} seconds".format(training_time))
    # print('returns:',returns)
    # plt.plot(returns)
    # Plot and save the first set of plots
    plt.plot(ep_return, label="training episode return")
    plt.plot(tot_rewards, label="MC eval of total reward")
    plt.legend()
    plt.savefig('train_plot.png')  # Save the first plot
    plt.close()  # Close the current figure before creating the next one

    # Plot and save the second set of plots
    plt.plot(disc_rewards, label="MC eval of discounted reward")
    plt.plot(V0, label="average $max_a Q(s_0)$")
    plt.legend()
    plt.savefig('monitoring_plot.png')  # Save the second plot
    plt.close()  # Close the current figure
    # Save trained agent
    agent.save()