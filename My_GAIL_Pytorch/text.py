import numpy as np
import torch

# a = np.array([1,2,2,2,2,2,2])
# b = np.array([1,2,2,2,2,2,2])
# print(id(a),"\n",id(b))
# a = np.append(a, b, axis=0)
# # a = np.concatenate((a, b), axis=1)
# print(id(a))
# print(a)
#
# c = torch.tensor([1, 2, 3, 4, 5])
# print(id(c))
# print("c:",c,"\n",c.numpy())
# print(id(c))

# t_1 = torch.rand(2, 5)
# t_2 = torch.rand(2, 5)
# c_1 = torch.cat([t_1, t_2], 0)
# c_2 = torch.cat([t_1, t_2], 1)
# print(c_1, "\n", c_2)


# c = torch.tensor([[1], [2]])
# b = a * -1
# d = a * c
# print("b:", b)
# print("d:", d)
# f = np.array([1, 2, 3, 4, 5])
# g = np.reshape(f, (5, 1))
# e = torch.from_numpy(g)
# print("f:", f)
# print("g:", g)
# print("e:", e)
# h = np.random((5,1))
# a = np.array([[1, 2, 3, 4, 5],
#                   [1, 2, 3, 4, 5]])
# print(a[1][1])
# a = torch.tensor([1, 2, 3, 4])
# b = torch.tensor([a for _ in range(len(a))]).to(torch.float32)
# a1 = a[1:]
# # print(a1)
# b1 = np.array(b[1:] > 1).mean()
# print(b1)
#
# b[2] = 8
# print(b)
# d = torch.mean(b)
# print(d, "\n", d.item())
# pass

import numpy as np
import torch

# def mem():
#     import os, psutil
#     return psutil.Process(os.getpid()).memory_info().rss // 1024
#
#
# # a = np.zeros(10000, dtype=np.float32)  # <-- 1. dtype is important
# # a = list(a)  # <-- 2. has to be a python list
#
# b = np.array([1, 2, 3])
# c = torch.as_tensor([b for _ in range(len(b))])
# d = torch .tensor([3, 2, 1], device=torch.device("cuda:0"))
#
# mem_base = mem()
# for j in range(20):
#     print("-"*20, j)
#     c = torch.as_tensor([b for _ in range(len(b))])
#     print("c.shape:", c.size())
#     for i in range(len(b)):
#         # torch.tensor(a)  # <-- 3. must not pass `dtype=torch.float32` here
#         c[i] = d
#         print(mem() - mem_base)

# 加 memory， 加 value 部分， 缩小discriminator的学习率
# 加 归一化， 鉴别器专家样本label加噪声


import random

#
# c = []
# for i in range(3):
#     c.append(None)
# a = [torch.tensor([1, 2, 3]), torch.tensor([10])]
# b = [torch.tensor([4, 5, 6]), torch.tensor([10])]
# d = [torch.tensor([7, 8, 9]), torch.tensor([10])]
# c[0] = a
# c[1] = b
# c[2] = d
#
# buffer = random.sample(c, 2)
# g = []
# print("1:", buffer, "buffer.shape:", len(buffer))
# for i in range(2):
#     g.append(buffer[i][0])
#     print("buffer[i][0]", buffer[i][0])
# g = torch.as_tensor(g)
# # map()
# print("done!")
# a = torch.tensor([1e-1])
# b = -torch.log(a) + 1
# print(b.item())
# c=round(0.123456789, 2)
# import datetime
# from torch.utils.tensorboard import SummaryWriter
# # Tensorboard
# writer = SummaryWriter(
#     'runs/{}_SAC111'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# writer.add_scalar("verified_episode_return", c, c)
#
# print(c)

a = torch.tensor([1e-2])
b = -torch.log(a)
print(b.item())
# a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [6, 7, 8, 9, 10]])
# print(a)
# b = a[2:2+2]
# print(b)
