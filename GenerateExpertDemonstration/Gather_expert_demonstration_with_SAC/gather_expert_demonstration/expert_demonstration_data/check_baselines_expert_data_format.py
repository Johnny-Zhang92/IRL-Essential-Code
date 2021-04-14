import numpy as np
import gym
import os.path as osp

# ex_path = 'deterministic.trpo.Hopper.0.00.npz'
# ex_data = np.load(ex_path)
# a = {'a':1, 'b':2, 'c':3}
# print(a)
# print(ex_data.files)
# print(ex_data.f)
# print(ex_data)
#
env = gym.make('Hopper-v2')
print(env.spec.id)

# Assemble the file name
file_path = 'model_guide/'
file_name = 'deterministic' + '_SAC_' + env.spec.id + '_johnny'
path_model_guide = osp.join(file_path, file_name)

# save expert data for contrast algorithm sam
# Assemble the file name
file_path = 'sam/'
file_name = 'deterministic' + '_SAC_' + env.spec.id + '_sam'
path_sam = osp.join(file_path, file_name)
# load expert data
model_guide_ex_data = np.load(path_model_guide + '.npz')
sam_ex_data = np.load(path_sam + '.npz')
sam_ex_data_items = sam_ex_data.items()
model_guide_returns = model_guide_ex_data['returns']
model_guide_returns1 = model_guide_returns[0:1]
model_guide_std = np.std(model_guide_returns)

data_map, stat_map = {}, {}
for k, v in sam_ex_data.items():
    if k in ['ep_env_rets', 'ep_lens']:
        stat_map[k] = v
    elif k in ['obs0', 'acs', 'env_rews', 'dones1', 'obs1']:
        data_map[k] = np.array(np.concatenate((v[:10])))

        # fmtstr = "[DEMOS] >>>> extracted {} transitions, from {} trajectories"
        # logger.info(fmtstr.format(len(self), self.num_demos))
        # rets_, lens_ = self.stat_map['ep_env_rets'], self.stat_map['ep_lens']
        # logger.info("  episodic return: {}({})".format(np.mean(rets_), np.std(rets_)))
        # logger.info("  episodic length: {}({})".format(np.mean(lens_), np.std(lens_)))



print('1')