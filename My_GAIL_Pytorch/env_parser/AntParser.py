import argparse


def get_parser():
    # Global Arguments
    parser = argparse.ArgumentParser(description='PyTorch GAIL Args')
    parser.add_argument('--env_name', type=str, default="Hopper-v2",
                        help="Environment Name")
    parser.add_argument('--seed', type=int, default=12345,
                        help="Algorithm Global Seed")
    parser.add_argument('--expert_path', type=str, default="./expert_demonstrations/deterministic_SAC_Hopper-v2_johnny.npz",
                        help="The path to save expert demonstrations")
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='run on CUDA (default: False)')
    parser.add_argument('--GPU_num', type=int, default=0,
                        help="Type the number of the GPU of your GPU machine you want to use if possible")
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automatically adjust α (default: False)')

    parser.add_argument('--num_steps', type=float, default=5e6, metavar='N',
                        help='maximum number of steps (default: 10000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--gen_replay_size', type=int, default=20000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--dis_replay_size', type=int, default=100, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument("--traj_limitation", type=int, default=-1,
                        help='the trajectory number in expert demonstration ')
    # Discriminator argument
    parser.add_argument("--dis_hidden_dim", type=int, default=100,
                        help="The hidden dimension of discriminator")
    parser.add_argument('--dis_rms_lr', type=float, default=0.00012, metavar='G',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dis_rms_alpha', type=float, default=0.99, metavar='G',
                        help='eps (default: 1e-9)')
    parser.add_argument('--dis_rms_eps', type=float, default=1e-8, metavar='G',
                        help='eps (default: 1e-9)')
    parser.add_argument('--dis_rms_weight_decay', type=float, default=2e-5, metavar='G',
                        help='weight_decay (default: 1e-4)')
    parser.add_argument('--dis_rms_momentum', type=float, default=0, metavar='G',
                        help='weight_decay (default: 1e-4)')

    # GAIL argument
    parser.add_argument('--segment_len', type=int, default=500, metavar='G',
                        help='The length of generated episode segment')
    parser.add_argument('--expert_batch', type=int, default=256, metavar='G',
                        help='Expert batch size')
    parser.add_argument('--g_steps', type=int, default=10, metavar='G',
                        help='The numbers to train generator')
    parser.add_argument('--d_steps', type=int, default=1, metavar='G',
                        help='The numbers to train discriminator')
    parser.add_argument('--generator_replay_batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--discriminator_replay_batch_size', type=int, default=1, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--dis_upd_exp_len', type=int, default=256, metavar='N',
                        help='the batch size to update discriminator by expert data; 0.4*segment_len')
    parser.add_argument('--dis_upd_gen_len', type=int, default=256, metavar='N',
                        help='the batch size to update discriminator by generated data;0.8*expert_batch')
    parser.add_argument('--dis_interval', type=int, default=3, metavar='N',
                        help='update discriminator net every dis_interval iters')
    parser.add_argument('--reward_baseline', type=float, default=1e-3, metavar='N',
                        help='The reward largest reward what is given by discriminator')
    parser.add_argument('--is_normal', type=int, default=0, metavar='N',
                        help='Whether using batch normal trick?')
    parser.add_argument('--scrip_num', type=int, default=9,
                        help='the num')
    args = parser.parse_args()
    return args

