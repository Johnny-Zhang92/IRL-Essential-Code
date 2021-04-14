import numpy as np
import torch

from common import logger


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.num_pairs = len(inputs)
        self.inputs = inputs
        self.labels = labels
        self.state = None
        self.next_state = None
        self.action = None
        # self.real_reward = rets
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        # print("self.num_pairs_init_:", self.num_pairs)
        # print("self.state_init_:", self.state.shape)
        # print("self.next_state_init_:", self.next_state.shape)
        # print("self.action_init_:", self.action.shape)

        self.init_pointer()

        # a = self.next_state
        # print('a:', a.shape)
        # print('inputs', inputs.shape)
        # print('leninputs', len(inputs))
        # print('state:', self.state.shape)

    def init_pointer(self):
        self.pointer = 0
        idx = np.arange(self.num_pairs - 1)
        # print("idx:", idx)
        if self.randomize:
            np.random.shuffle(idx)
            idx1 = 1 + idx
            # print("idx:", idx)
            # print("idx_max:", np.max(idx))
            # print("idx1_max:", np.max(idx1))
            # print("self.state_init_pointer1:", self.state.shape)
            # print("self.next_state_init_pointer1:", self.next_state.shape)
            # print("self.action_init_pointer1:", self.action.shape)
            self.state = self.inputs[idx, :]
            self.next_state = self.inputs[idx1, :]
            self.action = self.labels[idx, :]
            # print("self.state_init_pointer0:", self.state.shape)
            # print("self.next_state_init_pointer0:", self.next_state.shape)
            # print("self.action_init_pointer0:", self.action.shape)

        else:
            idx1 = 1 + idx
            print("idx1_random:", idx1)
            self.next_state = self.next_state[idx1, :]
            '''
            print('idx', idx.shape, '\n', 'idx1:', idx1.shape, '\n', 'self.state.shape', self.state.shape,
                  '\n', 'self.nexi_state', self.next_state.shape)          
                    import numpy as np           
                    a = [1, 2, 3, 4, 5]
                    a = np.vstack(a)
                    b = [6, 7, 8, 9, 10]
                    b = np.vstack(b)
                    idx = np.arange(5)
                    print("idx:", idx)
                    np.random.shuffle(idx)
                    print("shuffle idx:", idx)  
                    a1 = a[idx, :]
                    b1 = b[idx, :]
                    print("a:",a, "\n", "a1:", a1)
                    print("b:",b, "\n", "b1:", b1)

                    idx: [0 1 2 3 4]
                    shuffle idx: [4 2 0 3 1]
                    a: [[1]
                     [2]
                     [3]
                     [4]
                     [5]] 
                     a1: [[5]
                     [3]
                     [1]
                     [4]
                     [2]]
                    b: [[ 6]
                     [ 7]
                     [ 8]
                     [ 9]
                     [10]] 
                     b1: [[10]
                     [ 8]
                     [ 6]
                     [ 9]
                     [ 7]]
                    '''

    # convert to tensor
    def convert_to_tensor(self, state_batch, action_batch, next_state_batch):
        state_batch = torch.from_numpy(state_batch).to(torch.float32)
        action_batch = torch.from_numpy(action_batch).to(torch.float32)
        next_state_batch = torch.from_numpy(next_state_batch).to(torch.float32)
        # double
        return state_batch, action_batch, next_state_batch

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            idx1 = np.arange(self.num_pairs - 1)
            self.next_state = self.next_state[idx1, :]
            # state, action, next_state = \
            #     self.convert_to_tensor(self.state, self.action, self.next_state)
            return self.state, self.action, self.next_state
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        state = self.state[self.pointer:end, :]
        next_state = self.next_state[self.pointer:end, :]
        action = self.action[self.pointer:end, :]
        self.pointer = end
        # state, action, next_state = \
        #     self.convert_to_tensor(state, action, next_state)
        return state, action, next_state


class MuJoCoExpertData:
    def __init__(self, expert_path, train_fraction=0.7,  # default: train_fraction=0.7
                 traj_limitation=-1, randomize=True, log=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])  # number of trajectory 1500
            # print("traj_limitation:", traj_limitation)
        state = traj_data['obs'][:traj_limitation]  # store state, shape=(1500,1000,11),
        # 1000:the length of each  trajectory
        action = traj_data['acs'][:traj_limitation]  # store action, shape=(1500,1000,3)

        # env_name: Hopper
        # obs.shape:(1500,1000,11) acs.shape:(1500,1000,3)
        # ep_rets.shape:(1500) rews.shape:(1500,1000)
        # 1500:number of episodes, 1000:episode length 11:state space 3:action space
        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(state.shape) > 2:
            self.state = np.reshape(state, [-1, np.prod(state.shape[2:])])
            self.action = np.reshape(action, [-1, np.prod(action.shape[2:])])
        else:
            self.state = np.vstack(state)
            self.action = np.vstack(action)
        # Calculate the mean and std of state and action in numpy array format and convert to tensor format
        self.state_mean = torch.FloatTensor(self.state.mean(axis=0))
        self.state_std = self.state.std(axis=0)
        state_std = self.state_std.tolist()
        for i in range(len(state_std)):
            if state_std[i] == 0.0:
                state_std[i] = 1
        # self.state_std = np.array(state.std)
        self.state_std = torch.tensor(state_std)
        self.action_mean = torch.FloatTensor(self.action.mean(axis=0))
        self.action_std = torch.FloatTensor(self.action.std(axis=0))
        # print("self.state_mean:{},\nself.state_std:{}".format(self.state_mean, self.state_std))
        # print("self.action_mean:{},\nself.action_std:{}".format(self.action_mean, self.action_std))
        # the return of each trajectory
        # self.rets = traj_data['ep_rets'][:traj_limitation] # expert data made by openai baselines
        self.rets = traj_data['returns'][:traj_limitation]  # expert data made by my self
        self.avg_ret = sum(self.rets) / len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
       # print("len action:", len(self.action))
        # if len(self.action) > 2:
        #     self.action = np.squeeze(self.action)
        assert len(self.state) == len(self.action)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.state)
        self.randomize = randomize
        self.expert_demon = Dset(self.state, self.action, self.randomize)
        # for pretrain dynamic network
        self.train_set = Dset(self.state[:int(self.num_transition * train_fraction), :],
                              self.action[:int(self.num_transition * train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.state[int(self.num_transition * train_fraction):, :],
                            self.action[int(self.num_transition * train_fraction):, :],
                            self.randomize)
        if log:
            self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_expert_batch(self, batch_size, split=None):
        if split is None:
            return self.expert_demon.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()



def test(expert_path, traj_limitation, plot):
    dset = MuJoCoExpertData(expert_path, traj_limitation=traj_limitation)
    print("--------------------{-1}-----------------------")
    for i in range(int(1e4)):
        print("--------------------{}-----------------------".format(i))

        state1, action1, next_state1 = dset.get_next_expert_batch(batch_size=128, split='train')
        print("self.state_next_batch:", state1.shape)
        print("self.next_next_batch:", next_state1.shape)
        print("self.action_next_batch:", action1.shape)
    print('state1:\n', state1)
    # print('action1:\n', action1)
    # print('next_state1:\n', next_state1)
    if plot:
        dset.plot()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
