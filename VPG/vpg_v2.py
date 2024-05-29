'''
    Vanila Policy Gradient
    with rewards-to-go
'''

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = 32

        self.layer_1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.activation_function = nn.ReLU()
        self.layer_2 = nn.Linear(self.hidden_layer_size, self.output_size)

    def forward(self, obs):
        x = obs
        x = self.layer_1(x)
        x = self.activation_function(x)
        x = self.layer_2(x)
        return x

    def return_policy(self, obs):
        logits = self.forward(obs)
        return Categorical(logits=logits)

class vpg_agent:
    def __init__(self, obs_dim, action_dims):
        self.model = Model(obs_dim, action_dims)
        self.optimizer = Adam(self.model.parameters(), 1e-2)

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        policy = self.model.return_policy(obs_tensor)
        action = policy.sample()
        log_prob_action = policy.log_prob(action)
        action = action.item()
        return action, log_prob_action

    def calculate_loss(self, epoch_log_prob_actions, epoch_rewards_to_go):
        return -(epoch_log_prob_actions*epoch_rewards_to_go).mean()

    def train_one_epoch(self, env, max_timesteps, episode_timesteps):
        epoch_total_timesteps = 0
        epoch_returns = []
        epoch_log_prob_actions = []
        epoch_rewards_to_go = []

        while True:
            if epoch_total_timesteps > max_timesteps:
                break

            episode_reward = 0
            obs, _ = env.reset()
            curr_eps_rewards = []
            flag = 0

            for timestep in range(episode_timesteps):
                epoch_total_timesteps += 1
                action, log_prob_action = self.get_action(obs)
                obs, reward, done, trunc, _ = env.step(action)

                curr_eps_rewards.append(reward)
                episode_reward += reward
                epoch_log_prob_actions.append(log_prob_action)

                if done or trunc:
                    for i in range(timestep+1):
                        epoch_rewards_to_go.append(sum(curr_eps_rewards[i:]))
                    flag = 1
                    break

            if flag == 0:
                for i in range(timestep+1):
                    epoch_rewards_to_go.append(sum(curr_eps_rewards[i:]))

            epoch_returns.append(episode_reward)

        epoch_loss = self.calculate_loss(torch.stack(
            epoch_log_prob_actions), 
            torch.as_tensor(
                epoch_rewards_to_go, dtype=torch.float32)
        )
        epoch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.mean(epoch_returns)

def train():
    epochs = 1000
    max_timesteps = 5000
    episode_timesteps = 200
    env = gym.make("LunarLander-v2")
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n
    agent = vpg_agent(obs_dims, action_dims)

    for epoch in tqdm(range(epochs)):
        avg_return = agent.train_one_epoch(env, max_timesteps, episode_timesteps)
        print('epoch: %3d \t return: %.3f' % (epoch, avg_return))
        if (epoch+1)%10 == 0:
            vis_env = gym.make("LunarLander-v2", render_mode="human")
            done = False
            trunc = False
            obs, _ = vis_env.reset()
            while not done and not trunc:
                action, _ = agent.get_action(obs)
                obs, reward, done, trunc, _ = vis_env.step(action)
            vis_env.close()


if __name__ == '__main__':
    train()