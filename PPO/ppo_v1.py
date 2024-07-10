'''
    PPO with clip function, without entropy loss
'''

import numpy as np
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym
from tqdm import tqdm

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
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

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = 32
        self.loss = nn.MSELoss()

        self.layer_1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.activation_function = nn.ReLU()
        self.layer_2 = nn.Linear(self.hidden_layer_size, self.output_size)

    def forward(self, obs):
        x = obs
        x = self.layer_1(x)
        x = self.activation_function(x)
        x = self.layer_2(x)
        return x

class ppo_agent:
    def __init__(self, obs_dim, action_dims, gamma=0.99, alpha=3e-4,
        lambda_=0.95, policy_clip=0.2, batch_size=5, n_epochs=20):
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.actor = Actor(obs_dim, action_dims).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), 1e-2)
        self.critic = Critic(obs_dim, 1).to(self.device)
        self.critic_optimizer= Adam(self.critic.parameters(), 1e-2)
        self.gamma = gamma
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.lambda_ = lambda_
        self.policy_clip = policy_clip
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, obs):
        obs = T.tensor(obs).to(self.device)
        dist = self.actor.return_policy(obs)
        value = self.critic.forward(obs)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.lambda_
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.device)

            values = T.tensor(values).to(self.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)

                dist = self.actor.return_policy(states)
                critic_value = self.critic.forward(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

def train():
    n_games = 200
    N = 10
    n_steps = 0
    env = gym.make("LunarLander-v2")
    obs_dims = env.observation_space.shape[0]
    action_dims = env.action_space.n
    agent = ppo_agent(obs_dims, action_dims)

    for i in tqdm(range(n_games)):
        obs, _ = env.reset()
        done = False
        trunc = False
        rewards = []
        while not done and not trunc:
            action, probs, value = agent.choose_action(obs)
            obs_old = obs
            obs, reward, done, trunc, _ = env.step(action)
            rewards.append(reward)
            n_steps += 1
            agent.remember(obs_old, action, probs, value, reward, done)
            if n_steps % N == 0:
                agent.learn()
        print('iteration: %3d \t return: %.3f' % (i, np.sum(rewards)))
        if (i+1)%10 == 0:
            vis_env = gym.make("LunarLander-v2", render_mode="human")
            done = False
            trunc = False
            obs, _ = vis_env.reset()
            while not done and not trunc:
                action, probs, value = agent.choose_action(obs)
                obs, reward, done, trunc, _ = vis_env.step(action)
            vis_env.close()


if __name__ == '__main__':
    train()