import numpy as np
import matplotlib.pyplot as plt


returns_ppo = np.load('/home/zjh/桌面/bili/pingjun/0.8_DDPG_HalfCheetah.npy')
returns_trpo = np.load('/home/zjh/桌面/bili/pingjun/0.9_DDPG_HalfCheetah.npy')
returns_ddpg = np.load('/home/zjh/桌面/bili/pingjun/1_DDPG_HalfCheetah.npy')
returns_eerddpg = np.load('/home/zjh/桌面/最终实验/eerddpg_date/HalfCheetah.npy')

total1 = np.linspace(0, 1, len(returns_ppo))
total2 = np.linspace(0, 1, len(returns_trpo))
total3 = np.linspace(0, 1, len(returns_ddpg))
total4 = np.linspace(0, 1, len(returns_eerddpg))
fig = plt.figure()
plt.plot(total1, returns_ppo, label='0.7', color='lightgrey', linestyle='-', linewidth='1', marker='o', markevery=20)
plt.plot(total4, returns_eerddpg, label='0.8', color='black', linewidth='1')
plt.plot(total3, returns_ddpg, label='0.9', color='dimgrey', linestyle='--',  linewidth='1')
plt.plot(total2, returns_trpo, label='1', color='darkgrey', linestyle=':', linewidth='1')

plt.legend()
plt.xlabel('Time steps(1e6)')
plt.ylabel('Average Reward')
#plt.title('Humanoid-v2')
#plt.savefig('/home/zjh/桌面/new_shiyan/bili')
plt.show()