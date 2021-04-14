import gym
import numpy as np

#
# env = gym.make('Hopper-v2')
# a = env.spec.id + 'ssssss'
# # print(env.spec.id)
# print(a)

# a = [1,2,3,4,5,6,7,8,9,10]
# b = a[1:]
# print(b)
# b.append(None)
# print(b)
# b = np.array(b)
# print(b)

# a = np.ones((1, 5))
# print("a:\n", a)
# b = np.array([8])
# print("b:\n", b)
# a = np.append(a, [None])
# print("change a:\n", a)

# def reurns():
#     a, b = 1, 2
#     return a, b
# c = reurns()
# print(c)
# d, e = c
# print(d, e)
import torch
# a = torch.tensor([[1,2,3],[6,7,8111]])
# b = torch.tensor([1,2,3])
# c = a -b
# # print(c.size())
# expert_path = "./gather_expert_demonstration/expert_demonstration_data/model_guide/deterministic_SAC_InvertedDoublePendulum-v2_johnny.npz"
# expert_data = np.load(expert_path)
# a = expert_data.f.acs[:1,:]
# print(a)
# test_expert_path= "./gather_expert_demonstration/expert_demonstration_data/model_guide/deterministic_SAC_Hopper-v2_johnny.npz"
# expert_dataset = expert_data.MuJoCoExpertData(expert_path=test_expert_path,
#                                                   traj_limitation=-1)

 # scp -r ./deterministic_SAC_HalfCheetah-v2(6000)_johnny.npz aoty@192.168.126.133:/home/aoty/Model_guide_hopper/data/
env_name= "Hopper-v2"
env = gym.make(env_name)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.high)
print(env.action_space.low)