import argparse
import yaml


def get_global_passer():
    # Global Arguments
    parser = argparse.ArgumentParser(description='PyTorch GAIL Args')
    parser.add_argument('--env_name', type=str, default="Hopper-v2",
                        help="Environment Name")
    parser.add_argument('--seed', type=int, default=12345,
                        help="Algorithm Global Seed")
    parser.add_argument('--expert_path', type=str, default="None",
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

    parser.add_argument('--num_steps', type=int, default=1e7, metavar='N',
                        help='maximum number of steps (default: 10000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--gen_replay_size', type=int, default=50000, metavar='N',
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
    parser.add_argument('--expert_batch', type=int, default=128, metavar='G',
                        help='Expert batch size')
    parser.add_argument('--g_steps', type=int, default=50, metavar='G',
                        help='The numbers to train generator')
    parser.add_argument('--d_steps', type=int, default=1, metavar='G',
                        help='The numbers to train discriminator')
    parser.add_argument('--generator_replay_batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--discriminator_replay_batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--dis_upd_exp_len', type=int, default=8, metavar='N',
                        help='the batch size to update discriminator by expert data; 0.4*segment_len')
    parser.add_argument('--dis_upd_gen_len', type=int, default=500, metavar='N',
                        help='the batch size to update discriminator by generated data;0.8*expert_batch')
    parser.add_argument('--dis_interval', type=int, default=5, metavar='N',
                        help='update discriminator net every dis_interval iters')
    parser.add_argument('--reward_baseline', type=float, default=1e-3, metavar='N',
                        help='The reward largest reward what is given by discriminator')
    parser.add_argument('--reward_baseline1', type=float, default=1e-1, metavar='N',
                        help='The reward largest reward what is given by discriminator')
    args = parser.parse_args()
    return args


def get_env_args(args, env_yaml_path):
    # env Arguments
    variables = yaml.safe_load(open(env_yaml_path))
    args.expert_path = variables["expert_path"]
    # Generator parameter
    args.policy = variables["policy"]  # policy Type
    args.gamma = float(variables["gamma"])  # 'discount factor for reward (default: 0.99)'
    args.tau = float(variables["tau"])  # 'target smoothing coefficient(τ) (default: 0.005)'
    args.lr = float(variables["lr"])  # 'learning rate (default: 0.0003)'
    args.alpha = float(variables["alpha"])  # 'Temperature parameter α determines the relative importance of the
    #  entropy term against the reward (default: 0.2)'
    args.num_steps = float(variables["num_steps"])  # maximum number of steps
    args.gen_replay_size = variables["gen_replay_size"]  # size of replay buffer for training generator
    # 　Discriminator parameter
    args.dis_replay_size = float(variables["dis_replay_size"])  # size of replay buffer for training discriminator
    args.dis_rms_lr = float(variables["dis_rms_lr"])  # learning rate (default: 0.0005)
    args.dis_rms_alpha = float(variables["dis_rms_alpha"])  # eps (default: 1e-9)
    args.dis_rms_eps = float(variables["dis_rms_eps"])  # eps (default: 1e-9)
    args.dis_rms_weight_decay = float(variables["dis_rms_weight_decay"])  # weight_decay (default: 1e-4)
    args.dis_rms_momentum = float(variables["dis_rms_momentum"])  # weight_decay (default: 1e-4)
    # GAIL　parameter
    # the batch is pushed in replay buffer to train generator
    args.segment_len = variables["segment_len"]  # The length of generated episode segment in one iter
    args.expert_batch = variables["expert_batch"]  # The length of expert demonstrations batch in one iter
    # the steps to train generator and discriminator in one iter
    args.g_steps = variables["g_steps"]  # The update number to train generator per iter
    args.d_steps = variables["d_steps"]  # The update number to train discriminator per iter
    args.generator_replay_batch_size = variables[
        "generator_replay_batch_size"]  # the batch size to replay from generator memory
    args.discriminator_replay_batch_size = variables[
        "discriminator_replay_batch_size"]  # the batch size to replay from discriminator memory
    args.dis_upd_exp_len = variables["dis_upd_exp_len"]  # the batch size of generator data to update discriminator
    args.dis_upd_gen_len = variables["dis_upd_gen_len"]  # the batch size to expert data to update discriminator
    args.dis_interval = int(variables["dis_interval"])  # update discriminator net every dis_interval iters
    args.reward_baseline = float(
        variables["reward_baseline"])  # The reward largest reward what is given by discriminator
    args.reward_baseline1 = float(
        variables["reward_baseline1"])  # The reward largest reward what is given by discriminator
    args.is_normal = bool(variables["is_normal"])  # Whether using batch normal trick?
    return args
