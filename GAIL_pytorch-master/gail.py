from replay_buffer import replay_buffer
from net import disc_policy_net, value_net, discriminator, cont_policy_net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import gym
import random


class gail(object):
    def __init__(self, env, episode, capacity, gamma, lam, is_disc, value_learning_rate, policy_learning_rate, discriminator_learning_rate, batch_size, file, policy_iter, disc_iter, value_iter, epsilon, entropy_weight, train_iter, clip_grad, render):
        self.env = env
        self.episode = episode
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.is_disc = is_disc
        self.value_learning_rate = value_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.batch_size = batch_size
        self.file = file
        self.policy_iter = policy_iter
        self.disc_iter = disc_iter
        self.value_iter = value_iter
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.train_iter = train_iter
        self.clip_grad = clip_grad
        self.render = render

        self.observation_dim = self.env.observation_space.shape[0]
        if is_disc:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
        if is_disc:
            self.policy_net = disc_policy_net(self.observation_dim, self.action_dim)
        else:
            self.policy_net = cont_policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.discriminator = discriminator(self.observation_dim + self.action_dim)
        self.buffer = replay_buffer(self.capacity, self.gamma, self.lam)
        self.pool = pickle.load(self.file)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_learning_rate)
        self.disc_loss_func = nn.BCELoss()
        self.weight_reward = None
        self.weight_custom_reward = None

    def ppo_train(self, ):
        observations, actions, returns, advantages = self.buffer.sample(self.batch_size)
        observations = torch.FloatTensor(observations)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = advantages.detach()
        returns = torch.FloatTensor(returns).unsqueeze(1).detach()

        for _ in range(self.value_iter):
            values = self.value_net.forward(observations)
            value_loss = (returns - values).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        if self.is_disc:
            actions_d = torch.LongTensor(actions).unsqueeze(1)
            old_probs = self.policy_net.forward(observations)
            old_probs = old_probs.gather(1, actions_d)
            dist = torch.distributions.Categorical(old_probs)
            entropy = dist.entropy().unsqueeze(1)
            for _ in range(self.policy_iter):
                probs = self.policy_net.forward(observations)
                probs = probs.gather(1, actions_d)
                ratio = probs / old_probs.detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
                policy_loss = policy_loss.mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
                self.policy_optimizer.step()
        else:
            actions_c = torch.FloatTensor(actions)
            old_dist = self.policy_net.get_distribution(observations)
            old_log_probs = old_dist.log_prob(actions_c)
            entropy = old_dist.entropy().unsqueeze(1)
            for _ in range(self.policy_iter):
                dist = self.policy_net.get_distribution(observations)
                log_probs = dist.log_prob(actions_c)
                ratio = torch.exp(log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
                policy_loss = policy_loss.mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
                self.policy_optimizer.step()

    def discriminator_train(self):
        expert_batch = random.sample(self.pool, self.batch_size)
        expert_observations, expert_actions = zip(* expert_batch)
        expert_observations = np.vstack(expert_observations)
        expert_observations = torch.FloatTensor(expert_observations)
        if self.is_disc:
            expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1)
            expert_actions = torch.zeros(self.batch_size, self.action_dim)
            expert_actions.scatter_(1, expert_actions_index, 1)
        else:
            expert_actions = torch.FloatTensor(expert_actions).unsqueeze(1)
        expert_trajs = torch.cat([expert_observations, expert_actions], 1)
        expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0)

        observations, actions, _, _ = self.buffer.sample(self.batch_size)
        observations = torch.FloatTensor(observations)
        if self.is_disc:
            actions_index = torch.LongTensor(actions).unsqueeze(1)
            actions_dis = torch.zeros(self.batch_size, self.action_dim)
            actions_dis.scatter_(1, actions_index, 1)
        else:
            actions_dis = torch.FloatTensor(actions)
        trajs = torch.cat([observations, actions_dis], 1)
        labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0)

        for _ in range(self.disc_iter):
            expert_loss = self.disc_loss_func(self.discriminator.forward(expert_trajs), expert_labels)
            current_loss = self.disc_loss_func(self.discriminator.forward(trajs), labels)

            loss = (expert_loss + current_loss) / 2
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

    def get_reward(self, observation, action):
        observation = torch.FloatTensor(np.expand_dims(observation, 0))
        if self.is_disc:
            action_tensor = torch.zeros(1, self.action_dim)
            action_tensor[0, action] = 1.
        else:
            action_tensor = torch.FloatTensor(action).unsqueeze(1)
        traj = torch.cat([observation, action_tensor], 1)
        reward = self.discriminator.forward(traj)
        reward = - reward.log()
        return reward.detach().item()

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            total_reward = 0
            total_custom_reward = 0
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                if not self.is_disc:
                    action = [action]
                next_obs, reward, done, _ = self.env.step(action)
                custom_reward = self.get_reward(obs, action)
                value = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                self.buffer.store(obs, action, custom_reward, done, value)
                total_reward += reward
                total_custom_reward += custom_reward
                obs = next_obs
                if self.render:
                    self.env.render()

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    if not self.weight_custom_reward:
                        self.weight_custom_reward = total_custom_reward
                    else:
                        self.weight_custom_reward = 0.99 * self.weight_custom_reward + 0.01 * total_custom_reward
                    if len(self.buffer) >= self.train_iter:
                        self.buffer.process()
                        self.discriminator_train()
                        self.ppo_train()
                        self.buffer.clear()
                    print('episode: {}  reward: {:.2f}  custom_reward: {:.3f}  weight_reward: {:.2f}  weight_custom_reward: {:.4f}'.format(i + 1, total_reward, total_custom_reward, self.weight_reward, self.weight_custom_reward))
                    break
