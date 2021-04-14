import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Discriminator_ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, discriminator_reward, label):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [discriminator_reward, label]
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        buffer = random.sample(self.buffer, 1)
        discriminator_reward = buffer[0][0]
        label = buffer[0][1]
        return discriminator_reward, label

    def __len__(self):
        return len(self.buffer)
