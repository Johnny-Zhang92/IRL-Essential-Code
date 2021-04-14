import torch
import torch.nn as nn
import torch.nn.functional as F
from common import logger, expert_data
from tqdm import tqdm
from torch.optim import rmsprop
import numpy as np


# Initialize neural network weights
def weights_init_(m):
    # torch.nn.init.normal_(m, mean=0, std=1)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        torch.nn.init.constant_(m.bias, 0)


'''
variables:
dynamic_number: the number of ensemble dynamic
s_a_s'(expert experience): experience 
num_state: state dimension 
num_action: action dimension
hidden_dim: hidden dimension of dynamic neural network
'''


# create neural network of dynamic
class DynamicNetwork(nn.Module):
    def __init__(self, num_state, num_action, dynamic_hidden_dim):
        super(DynamicNetwork, self).__init__()

        self.linear1 = nn.Linear(num_state + num_action, dynamic_hidden_dim)
        self.linear2 = nn.Linear(dynamic_hidden_dim, dynamic_hidden_dim)
        self.linear3 = nn.Linear(dynamic_hidden_dim, dynamic_hidden_dim)
        self.linear4 = nn.Linear(dynamic_hidden_dim, num_state)

        self.apply(weights_init_)

    # use dynamic neural network to compute next state s'
    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)
        return x1

    def to(self, device):
        return super(DynamicNetwork, self).to(device)

    # # compute dynamic model loss
    # def dynamic_loss(self, ):
    #     pass
    #
    # # update dynamic model
    # def update_dynamic(self, ):
    #     pass
    #
    # # compute reward
    # def get_reward(self):
    #     pass


'''
    # create neural network of dynamic
    def ensemble_dynamic(self, x, reuse):
        pass

    # extracted the s_a_s' from reply buffer one by one
    # def sample_replay_buffer(self, batch_size, updates=0):
    #     pass

    # pretrain dynamic model
    def pretrain_dymodel(self, expert_path, batch_size, updates=0):
        #
        expert_demon = expert_data.MuJoCoExpertData(expert_path, traj_limitation=traj_limitation)
        state, next_state, action = expert_demon.get_next_expert_batch(batch_size)

'''


class DynamicModel(nn.Module):
    def __init__(self, state_dim, action_dim, args, num_dynamic=1):
        super(DynamicModel, self).__init__()
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.dy_network = DynamicNetwork(state_dim, action_dim,
                                         args.dynamic_hidden_dim).to(self.device)
        self.num_dynamic = num_dynamic
        # self.dy_network_optim = Adam(self.dy_network.parameters(), lr=args.lr, eps=1e-9)  # lr:learning rate
        # self.dy_network_optim = adadelta.Adadelta(self.dy_network.parameters(),
        #                                           lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
        # self.dy_network_optim = SGD(self.dy_network.parameters(), lr=args.lr2)
        self.dy_network_optim = rmsprop.RMSprop(self.dy_network.parameters(), lr=args.dy_rms_lr,
                                                alpha=args.dy_rms_alpha, eps=args.dy_rms_eps,
                                                weight_decay=args.dy_rms_weight_decay,
                                                momentum=args.dy_rms_momentum)
        # self.expert_path = expert_data
        # self.traj_limination = traj_limitation

    def forward(self, state, action):
        x = self.dy_network(state.to(self.device), action.to(self.device))
        return x

    #  compute fix dynamic network loss
    def update_dynamic_parameters(self, state_batch, action_batch, next_state_batch, split=None):

        predict_next_state_batch = self.dy_network(state_batch.to(self.device),
                                                   action_batch.to(self.device))
        # loss function
        loss_function = nn.MSELoss()
        loss = loss_function(predict_next_state_batch, next_state_batch.to(self.device))
        if split == "val":
            return loss.item()
        self.dy_network_optim.zero_grad()  # clean gradient
        loss.backward()  # compute gradient
        self.dy_network_optim.step()  # update weight
        if split == "train":
            return loss.item()

    #  compute real dynamic network loss
    def update_real_dynamic_parameters(self, memory, batch_size):
        # sample from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, dy_reward_batch \
            = memory.sample(batch_size=batch_size)
        # convert to tensor
        state_batch = torch.from_numpy(state_batch).to(torch.float32)
        action_batch = torch.from_numpy(action_batch).to(torch.float32)
        next_state_batch = torch.from_numpy(next_state_batch).to(torch.float32)
        # compute loss
        train_loss = self.update_dynamic_parameters(state_batch, action_batch,
                                                    next_state_batch, split='train')
        return train_loss

    # pretrain dynamic model
    def pretrain_dynamic_model(self, expert_path, traj_limitation,
                               pre_train_batchsize=128, max_iters=1e5,
                               verbose=False):
        val_per_iter = int(max_iters / 10)
        dataset = expert_data.MuJoCoExpertData(expert_path=expert_path,
                                               traj_limitation=traj_limitation)
        # start pretraining
        logger.log("Pretraining Dynamic Network with Expert Demonstration ")
        for iter_so_far in tqdm(range(int(max_iters))):
            # get train bath_set from expert demonstration(.npz)
            state_batch, action_batch, next_state_batch = \
                dataset.get_next_expert_batch(pre_train_batchsize, 'train')
            # state action and next state normalization
            # state_batch_normal = (state_batch - dataset.state_mean.to("cuda")) / dataset.state_std.to("cuda")
            # next_state_batchstate_batch_normal = (next_state_batch - dataset.state_mean.to("cuda")) / \
            #                                      dataset.state_std.to("cuda")
            # update dynamic network parameters
            train_loss = self.update_dynamic_parameters(state_batch, action_batch,
                                                        next_state_batch, split='train')
            # verify dynamic network performance
            if verbose and iter_so_far % val_per_iter == 0:
                state_batch, action_batch, next_state_batch = \
                    dataset.get_next_expert_batch(-1, 'val')
                val_loss = self.update_dynamic_parameters(state_batch, action_batch, next_state_batch, split='val')
                logger.log("Training loss:{},Validation loss:{}".format(train_loss, val_loss))

    # verify dynamic model
    def verify_dy_model(self, expert_path, traj_limitation):

        dataset = expert_data.MuJoCoExpertData(expert_path=expert_path,
                                               traj_limitation=traj_limitation,
                                               log=False)
        # verify dynamic network performance
        state_batch, action_batch, next_state_batch = \
            dataset.get_next_expert_batch(-1, 'val')

        val_loss = self.update_dynamic_parameters(state_batch, action_batch,
                                                  next_state_batch, split='val')
        return val_loss


# verify dynamic model var reward
def verify_dy_model_reward(expert_path, traj_limitation, dynamic_models=None,
                           batch_size=128, dy_num=5):
    dataset = expert_data.MuJoCoExpertData(expert_path=expert_path,
                                           traj_limitation=traj_limitation,
                                           log=False)
    # get date from expert demonstration
    state_batch, action_batch, next_state_batch = \
        dataset.get_next_expert_batch(batch_size, 'val')
    predict_next_state = [dynamic_models[i](state_batch, action_batch) for i in range(dy_num)]
    # print('predict_next_state.size:', predict_next_state)
    dy_reward = np.empty([batch_size])
    # print('dy_reward.size:', dy_reward)
    for k in range(batch_size):
        var = []
        for i in range(state_batch.size(1)):
            var1 = []
            for j in range(dy_num):
                # print('predict_next_state', predict_next_state)
                # print('predict_next_state[j][i]', predict_next_state[j][0][i])
                # print('predict_next_state[j][i].item()', predict_next_state[j][0][i].item())
                var1.append(predict_next_state[j][k][i].item())

            var1 = torch.from_numpy(np.array(var1)).to(torch.float64)
            # print('var1:', var1)
            var.append(torch.var(var1).item())
        # print('var:', var)
        var_reward = np.mean(var)
        # print('var_reward:', var_reward)
        dy_reward[k] = var_reward
    return dy_reward


# use ensemble dynamic variation to compute reward
def get_dynamic_reward(dynamic_models, state, action,
                       reward_threshold, batch_size=1, dy_num=5):
    predict_next_state = [dynamic_models[i](state, action) for i in range(dy_num)]
    # print('predict_next_state.size:', predict_next_state)
    dy_reward = np.empty([batch_size])
    # print('dy_reward.size:', dy_reward)
    for k in range(batch_size):
        var = []
        for i in range(state.size(1)):
            var1 = []
            for j in range(dy_num):
                # print('predict_next_state', predict_next_state)
                # print('predict_next_state[j][i]', predict_next_state[j][0][i])
                # print('predict_next_state[j][i].item()', predict_next_state[j][0][i].item())
                var1.append(predict_next_state[j][k][i].item())

            var1 = torch.from_numpy(np.array(var1)).to(torch.float64)
            # print('var1:', var1)
            var.append(torch.var(var1).item())
        # print('var:', var)
        var_reward = np.mean(var)
        # print('var_reward:', var_reward)
        # var_reward = torch.var(predict_next_state)  # compute  var(fang cha)
        # compute reward
        # print('var_reward_front', var_reward)
        # print('reward_threshold', var_reward)
        if var_reward > reward_threshold:
            var_reward = 0
        else:
            var_reward = 1
        # print('var_reward_after', var_reward)
        dy_reward[k] = var_reward
    return dy_reward  # numpy array, size(batch_size)


# use ensemble dynamic variation to compute reward
# def get_dynamic_reward2(dynamic_models, state, action,
#                        reward_threshold, batch_size=1, dy_num=5):
#     batch_dy_models_state = [] # shape: batch_size
#     for i in range(batch_size):
#         for j in range(dy_num):
#             # dynamic_models[j](state, action) # shape:batch_size*state_dim
#                 for k in range(batch_size):
#                     pass


def create_one_dy_model(args):
    # traj_data = np.load(args.expert_path)
    # state_dim = np.prod(traj_data['obs'].shape[2:])  # state.shape=(1500,1000,11),
    # action_dim = np.prod(traj_data['acs'].shape[2:])
    import gym
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    pre_train = DynamicModel(state_dim, action_dim, args)
    pre_train.pretrain_dynamic_model(args.expert_path, args.traj_limitation, pre_train_batchsize=args.pre_train_batch,
                                     max_iters=args.max_iters, verbose=True)
    # verify real dynamic network loss
    total_ver_loss = 0
    verify_loss = []
    print("verifying dy_models_real loss")
    for _ in range(10):
        verify_loss.append(DynamicModel.verify_dy_model(args.expert_path, args.traj_limitation))
    mean_ver_loss = np.mean(verify_loss)
    print("verify_loss of dy_models_real is {}".format(verify_loss))


def create_ensemble_dy_model(train,  verify_var_and_loss, args):
    import gym
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # create DynamicModel class
    dy_models = [DynamicModel(state_dim, action_dim, args) for _ in range(args.dynamic_num)]
    dy_models_real = DynamicModel(state_dim, action_dim, args)
    if train:
        # training the dynamic net one by one
        for i in range(args.dynamic_num):
            print("-----------------pretraining dy_models[{}]------------------------".format(i))
            dy_models[i].pretrain_dynamic_model(args.expert_path, args.traj_limitation,
                                                max_iters=args.max_iters, verbose=True)
        print("-----------------pretraining dy_models_real------------------------")
        dy_models_real.pretrain_dynamic_model(args.expert_path, args.traj_limitation,
                                              max_iters=args.max_iters, verbose=True)
        for i in range(args.dynamic_num):
            dy_models_path = "models/{}/dy_models[{}]_{}".format(args.env_name, i, args.env_name)
            torch.save(dy_models[i].state_dict(), dy_models_path)
            print('Saving dy_models to {}'.format(dy_models_path))
        dy_models_real_path = "models/{}/dy_models_real_{}".format(args.env_name, args.env_name)
        print('Saving dy_models_real to {}'.format(dy_models_real_path))
        torch.save(dy_models_real.state_dict(), dy_models_real_path)
    if train or verify_var_and_loss:
        for i in range(args.dynamic_num):
            dy_models_path = "models/{}/dy_models[{}]_{}".format(args.env_name, i, args.env_name)
            print('Loading dy_models from {}'.format(dy_models_path))
            dy_models[i].load_state_dict(torch.load(dy_models_path))
        dy_models_real_path = "models/{}/dy_models_real_{}".format(args.env_name, args.env_name)
        print('Loading dy_models_real from {}'.format(dy_models_real_path))
        dy_models_real.load_state_dict(torch.load(dy_models_real_path))
        if args.verify_dynamic_loss:
            # verify dynamic network loss
            for i in range(args.dynamic_num):
                total_ver_loss = 0
                verify_loss = []
                print("verifying dy_model[{}] loss".format(i))
                for j in range(5):
                    verify_loss.append(dy_models[i].verify_dy_model(args.expert_path, args.traj_limitation))
                mean_ver_loss = np.mean(verify_loss)
                print("verify_loss of dy_model[{}] is {}".format(i, verify_loss))
                print("mean_ver_loss of dy_model[{}] is {}".format(i, mean_ver_loss))
            # verify real dynamic network loss
            total_ver_loss = 0
            verify_loss = []
            print("verifying dy_models_real loss")
            for _ in range(10):
                verify_loss.append(dy_models_real.verify_dy_model(args.expert_path, args.traj_limitation))
            mean_ver_loss = np.mean(verify_loss)
            print("verify_loss of dy_models_real is {}".format(verify_loss))
            print("mean_ver_loss of dy_models_real is {}".format(mean_ver_loss))
            # calculate dynamic model var reward
        for _ in range(5):
            dy_reward = verify_dy_model_reward(expert_path=args.expert_path,
                                               traj_limitation=args.traj_limitation,
                                               dynamic_models=dy_models,
                                               batch_size=args.model_sample_batch_size,
                                               dy_num=args.dynamic_num)
            # print("calculate dynamic model var reward:", dy_reward)
            print("dynamic model var  shape:", dy_reward.shape, '\n', "dy_var.max:", dy_reward.max(),
                  '\n', "dy_var.min:", dy_reward.min(), '\n', "dy_var.mean:", dy_reward.mean())

# compute on GPU
# ensemble
# check point save network
#


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # dynamic network hidden dimension
    parser.add_argument("--dynamic_hidden_dim", type=int, default=200)
    parser.add_argument("--cuda", type=bool, default=True)
    # lr = 3e-4 eps = 1e-8
    parser.add_argument('--lr', type=float, default=8e-4, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--eps', type=float, default=1e-7, metavar='G',
                        help='eps (default: 1e-9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='G',
                        help='weight_decay (default: 1e-4)')
    # RMSprop
    # default:lr = 1e-2, alpha = 0.99, eps = 1e-8, weight_decay = 0, momentum = 0
    parser.add_argument('--dy_rms_lr', type=float, default=0.0005, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--dy_rms_alpha', type=float, default=0.99, metavar='G',
                        help='eps (default: 1e-9)')
    parser.add_argument('--dy_rms_eps', type=float, default=1e-8, metavar='G',
                        help='eps (default: 1e-9)')
    parser.add_argument('--dy_rms_weight_decay', type=float, default=2e-5, metavar='G',
                        help='weight_decay (default: 1e-4)')
    parser.add_argument('--dy_rms_momentum', type=float, default=0, metavar='G',
                        help='weight_decay (default: 1e-4)')

    parser.add_argument('--pre_train_batch', type=int, default=128, metavar='G',
                        help='pre_train_batch (default: 128)')
    parser.add_argument('--max_iters', type=float, default=1e5, metavar='G',
                        help='max_iters (default: 1e5)')
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v2")
    parser.add_argument("--expert_path", type=str, default="data/deterministic_SAC_HalfCheetah-v2_johnny.npz")
    parser.add_argument("--traj_limitation", type=int, default=-1)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--model_sample_batch_size", type=int, default=32)
    parser.add_argument("--verify_dynamic_loss", type=bool, default=False)
    parser.add_argument("--dynamic_num", type=int, default=5)
    args = parser.parse_args()

    # create_one_dy_model(args)
    train = 1
    # train = 0
    verify_var_and_loss = 1
    # verify_var_and_loss = 0
    create_ensemble_dy_model(train, verify_var_and_loss, args)
