
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from passer import get_passer
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

# get args
args = get_passer()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make("Hopper-v2")
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

actor_path = "./models/Hopper/sac_actor_Hopper(2000)"
critic_path = "./models/Hopper/sac_critic_Hopper(2000)"
agent.load_model(actor_path, critic_path)

avg_reward = 0.
episodes = 30
episde_len = 0
for _ in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=False)

        next_state, reward, done, _ = env.step(action)
        env.render()
        episode_reward += reward
        episde_len += 1

        state = next_state
    avg_reward += episode_reward
    print("episode_return:", episode_reward, "episode_len:", episde_len)
avg_reward /= episodes
episode_reward1 = avg_reward
print("episode_reward1:", episode_reward1)
env.close()