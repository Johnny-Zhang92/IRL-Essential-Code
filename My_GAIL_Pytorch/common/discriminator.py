import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import rmsprop
from torch.distributions import Normal


# from torch.distributions import Categorical
# Categorical.entropy()


# Initialize neural network weights
def weights_init_(m):
    # torch.nn.init.normal_(m, mean=0, std=1)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        torch.nn.init.constant_(m.bias, 0)


class Discriminator_Net(nn.Module):
    def __init__(self, state_action_space, hidden_dim):
        super(Discriminator_Net, self).__init__()

        self.linear1 = nn.Linear(state_action_space, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear5 = nn.Sigmoid()

        self.apply(weights_init_)

    def forward(self, state_action):
        x1 = F.relu(self.linear1(state_action))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)
        x1 = self.linear5(x1)
        return x1


class DiscriminatorModel(object):
    def __init__(self, state_space, action_space, args):
        self.state_action_space = state_space + action_space
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.discriminator = Discriminator_Net(self.state_action_space, args.dis_hidden_dim).to(self.device)
        self.dis_optim = rmsprop.RMSprop(self.discriminator.parameters(), lr=args.dis_rms_lr,
                                         alpha=args.dis_rms_alpha, eps=args.dis_rms_eps,
                                         weight_decay=args.dis_rms_weight_decay,
                                         momentum=args.dis_rms_momentum)

    def update_discriminator(self, dsa, segment_len, exp_current_len=None, gen_current_len=None):
        # real_index = np.empty([len(mix_index), 1])
        # negative_real_index = np.empty([len(mix_index), 1])
        # for i in range(len(mix_index)):
        #     if mix_index[i] <= 99:
        #         real_index[i][1] = 1
        #         negative_real_index[i][1] = -1
        #     else:
        #         real_index[i][1] = 0
        #         negative_real_index[i][1] = 1
        #
        # real_index = torch.from_numpy(real_index.T)
        # negative_real_index = torch.from_numpy(negative_real_index.T)

        # label1 = np.zeros([segment_len, 1])
        # label2 = np.ones([expert_batch, 1])
        # label = torch.from_numpy(np.vstack((label1, label2))).to(self.device)
        # label1 = torch.zeros([segment_len, 1], device=self.device)
        # label2 = torch.ones([expert_batch, 1], device=self.device)
        # label = torch.cat([label1, label2], dim=0)
        # generator label
        noise1 = torch.ones([gen_current_len, 1], device=self.device, dtype=torch.float32)
        noise1 = noise1.normal_(0., std=0.10)
        noise1 = noise1.clamp(-0.15, 0.3)
        label1 = torch.zeros([gen_current_len, 1], device=self.device)
        # label1 = label1 + noise1 + 0.15
        # expert label
        noise2 = torch.ones([exp_current_len, 1], device=self.device, dtype=torch.float32)
        noise2 = noise2.normal_(0., std=0.10)
        noise2 = noise2.clamp(-0.05, 0.05)
        label2 = torch.ones([exp_current_len, 1], device=self.device)
        label2 = label2 * 0.9 + noise2
        label = torch.cat([label1, label2], dim=0)
        new_dsa = torch.cat([dsa[:gen_current_len], dsa[segment_len:segment_len + exp_current_len]], dim=0)

        dis_loss = torch.nn.BCELoss()
        loss = dis_loss(new_dsa, label)

        self.dis_optim.zero_grad()
        loss.backward()
        self.dis_optim.step()
        return loss.item()

    def train_discriminator(self, memory):
        # Sample a batch from memory
        dis_reward_batch, label_batch = memory.sample()

        dis_loss = torch.nn.BCELoss()
        loss = dis_loss(dis_reward_batch, label_batch)

        self.dis_optim.zero_grad()
        loss.backward()
        self.dis_optim.step()
        return loss.item()

    def pretrain_discriminator(self, dsa):
        pass
