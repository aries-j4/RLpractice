''' 
    Deep Q Network
'''

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

class DQNMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

        self.batch_size = batch_size

    def generate_batches(self):
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.rewards),\
                np.array(self.dones),\
                np.array(self.next_states)

    def store_memory(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.states[0]
        del self.actions[0]
        del self.rewards[0]
        del self.dones[0]
        del self.next_states[0]

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = 64 

        self.layer1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.activation_function = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.layer3 = nn.Linear(self.hidden_layer_size, self.output_size)

    def forward(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        x = self.layer1(obs_tensor)
        x = self.activation_function(x)
        x = self.layer2(x)
        x = self.activation_function(x)
        x = self.layer3(x)
        return x

class DQN_Agent:
    def __init__(self, obs_dims, action_dims, batch_size, gamma=0.99, epsilon=1.0, 
                eps_end=0.01, eps_dec=0.995, lr=5e-4, max_mem_size=1_00_000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.tau = 1e-3

        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.target_updation = 5
        self.target_update_counter = 0
        self.update_every = 4
        self.update_cntr = 0

        self.batch_size = batch_size
        self.obs_dims = obs_dims
        self.action_dims = action_dims

        self.model = DQN(self.obs_dims, self.action_dims).to(self.device)
        self.target = DQN(self.obs_dims, self.action_dims).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.memory = DQNMemory(self.batch_size)

    def remember(self, obs, action, reward, done, next_obs):
        self.memory.store_memory(obs, action, reward, done, next_obs)
        self.mem_cntr += 1
        if self.mem_cntr > self.mem_size:
            self.memory.clear_memory()
            self.mem_cntr -= 1
        self.update_cntr += 1
        if self.update_cntr > self.update_every:
            self.learn()
            self.update_cntr = 0

    def choose_action(self, obs):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_min else self.epsilon
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dims)
            return action
        obs = torch.tensor(obs).to(self.device)
        Qs = self.model.forward(obs)
        action = np.argsort(Qs.detach().cpu().numpy())
        action = action[0]
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.optimizer.zero_grad()

        state_arr, action_arr, reward_arr, dones_arr, next_state_arr = self.memory.generate_batches()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_state_arr[batch], dtype=torch.float).to(self.device)
        terminal_batch = torch.tensor(dones_arr[batch]).to(self.device)

        action_batch = np.array(action_arr[batch], np.int32)
        reward_batch = torch.tensor(reward_arr[batch]).to(self.device)
        
        q_eval = self.model.forward(states)[batch_index, action_batch]
        q_next = self.target.forward(next_states)[batch_index, action_batch]
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=0)[0]
        model_loss = F.mse_loss(q_eval.double(), q_target.double()).to(self.device)
        model_loss.backward()
        self.optimizer.step()

        # self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.epsilon

        self.target_update_counter += 1

        for target_param, local_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        # if self.target_update_counter > self.target_updation:
        #     # self.target.load_state_dict(self.tau*self.model.state_dict()+(1-self.tau)*self.target.state_dict())
        #     self.target_update_counter = 0

def train():
    n_games = 2000
    env = gym.make("LunarLander-v2")
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n
    agent = DQN_Agent(obs_dims, action_dims, 64)

    for i in tqdm(range(n_games)):
        obs, _ = env.reset()
        done = False
        trunc = False
        rewards = []
        while not done and not trunc:
            action = agent.choose_action(obs)
            obs_old = obs
            obs, reward, done, trunc, _ = env.step(action)
            rewards.append(reward)
            agent.remember(obs_old, action, reward, done, obs)
            # agent.learn()
        print('iteration: %3d \t return: %.3f' % (i, np.sum(rewards)))
        if (i+1)%10 == 0:
            vis_env = gym.make("LunarLander-v2", render_mode="human")
            done = False
            trunc = False
            obs, _ = vis_env.reset()
            while not done and not trunc:
                action = agent.choose_action(obs)
                obs, reward, done, trunc, _ = vis_env.step(action)
            vis_env.close()


if __name__ == '__main__':
    train()