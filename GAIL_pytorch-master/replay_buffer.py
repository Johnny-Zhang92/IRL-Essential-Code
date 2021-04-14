import numpy as np
import random
from collections import deque


class replay_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, done, value):
        observation = np.expand_dims(observation, 0)
        self.memory.append([observation, action, reward, done, value])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, dones, values, returns, advantages = zip(* batch)
        return np.concatenate(observations, 0), actions, returns, advantages

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[3]) + traj[4]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[2] + Value_previous * self.gamma * (1 - traj[3]) - traj[4]
            Adv = delta + (1 - traj[3]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[4]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()