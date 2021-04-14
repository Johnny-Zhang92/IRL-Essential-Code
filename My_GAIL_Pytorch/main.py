import torch
import numpy as np
import gym
from common import expert_data
import time
import csv
import datetime

from torch.utils.tensorboard import SummaryWriter
from common.discriminator import DiscriminatorModel
from common.sac import SAC
from env_parser import parser
from contextlib import contextmanager
from common.color_print import Colored
from common.replay_memory import ReplayMemory
from common.replay_memory import Discriminator_ReplayMemory


@contextmanager
def timed(msg):
    color_text = Colored()
    print(color_text.blue(msg))
    start_time = time.time()
    yield
    # print(color_text.blue("Taking {} seconds".format(time.time() - start_time)))


def get_env_parser(envs, global_args):
    args = None
    for env in envs:
        if env == global_args.env_name:
            env_yml_path = "env_parser/{}_variable.yaml".format(env)
            args = parser.get_env_args(global_args, env_yaml_path=env_yml_path)
    if args is None:
        print("No matching environment in list:environments，please add yaml file to path ./env_parser/,", "\n",
              "then add environment name to list environments")
        exit(1)
    return args


def generate_segment(env, actor, segment_len, device, normal, is_normal=True):
    total_steps, i = 0, 0
    action = env.action_space.sample()
    done = True
    state = env.reset()
    state_array = np.array([state for _ in range(segment_len)])
    next_state_array = np.array([state for _ in range(segment_len)])
    action_array = np.array([action for _ in range(segment_len)])
    # entropy_array = np.array([action for _ in range(segment_len)])
    entropy_tensor = torch.as_tensor([[1] for _ in range(segment_len)], dtype=torch.float32).to(device)
    mask_array = np.array([[1] for _ in range(segment_len)])
    episode_return, episode_len = 0, 0

    while True:
        if is_normal:
            state_normal = (state - normal[0]) / normal[1]
            action, entropy = actor.select_action(state_normal, evaluate=False)
            action_normal = (action - normal[2]) / normal[3]
        else:
            action, entropy = actor.select_action(state, evaluate=False)
        if total_steps >= 0 and total_steps % segment_len == 0:
            yield {"state": state_array,
                   "action": action_array,
                   "next_state": next_state_array,
                   "entropy": entropy_tensor,
                   # "entropy": entropy_array,
                   "mask": mask_array,
                   "total_steps": total_steps,
                   "episode_return": episode_return
                   }
            # entropy_tensor.detach()
            entropy_tensor = torch.as_tensor([action for _ in range(segment_len)], dtype=torch.float32).to(device)
            # entropy_tensor.requires_grad_(False)
            # entropy_tensor.requires_grad_(True)

            i = 0
            # action, entropy = actor.select_action(state, evaluate=True)

        state_array[i] = state
        action_array[i] = action
        # entropy_array[i] = entropy
        entropy_tensor[i] = torch.mean(entropy)  # 此处应改为tensor，不然生成器的loss函数无法求导，

        if is_normal:
            state, env_rew, done, _ = env.step(action_normal)
            # print("1")
        else:
            state, env_rew, done, _ = env.step(action)
            # print("2")
            # env.render()

        episode_return += env_rew
        episode_len += 1
        mask_array[i] = 1 if episode_len == env._max_episode_steps else float(not done)
        next_state_array[i] = state
        i += 1

        if done:
            episode_return, episode_len = 0, 0
            state = env.reset()
        total_steps += 1


def train_gail(args):
    # Tensorboard
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}_seed[{}]-[]'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                            args.policy, args.seed, args.scrip_num))

    # Environment
    env = gym.make(args.env_name)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Define actor, using SAC　generator
    generator = SAC(env.observation_space.shape[0], env.action_space, args)

    # Load expert data
    expert_dataset = expert_data.MuJoCoExpertData(expert_path=args.expert_path,
                                                  traj_limitation=args.traj_limitation)
    state_mean, state_std, action_mean, action_std = expert_dataset.state_mean.numpy(), expert_dataset.state_std.numpy(), \
                                                     expert_dataset.action_mean.numpy(), expert_dataset.action_std.numpy()

    # Define replay buffer
    generator_memory = ReplayMemory(args.gen_replay_size, args.seed)
    # discriminator_memory = Discriminator_ReplayMemory(args.dis_replay_size, args.seed)

    # pretrained by BC algorithm
    # if args.pretrain_bc:
    #     import behavior_clone
    #     generator.behavior_clone()

    # Define discriminator
    discriminator = DiscriminatorModel(env.observation_space.shape[0], env.action_space.shape[0], args)

    # Define the class what generates a segment
    normal_list = [state_mean, state_std, action_mean, action_std]

    segment_generator = generate_segment(env, generator, args.segment_len, device, normal=normal_list,
                                         is_normal=args.is_normal)

    # pretrain the discriminator
    # if args.pretrain:
    #     # sampling expert data set from expert demonstrations, type: torch tensor, device:cpu
    #     expert_state, expert_action, expert_next_state = expert_dataset.get_next_expert_batch(args.expert_batch)
    #     total_data = np.concatenate((expert_state, expert_action))
    #     discriminator_reward = discriminator.discriminator(
    #         torch.from_numpy(total_data).to(torch.float32).to(device))
    #     for i in range(args.d_steps):
    #         discriminator.pretrain_discriminator(dsa=discriminator_reward, expert_batch=args.expert_batch)
    #
    #     if len(memory) < args.replay_size:
    #         for i in range(args.expert_batch):
    #             memory.push(expert_state[i], expert_action["action"][i], 1,
    #                         expert_next_state["next_state"][i], 1.0)
    #     for _ in range(args.g_steps):
    #         critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = generator.update_parameters(memory,
    #                                                                                                  args.batch_size,
    #                                                                                                  0)

    color_text = Colored()
    avg_reward1 = 0
    iters, sample_interval, print_interval, writer_interval = 0, 1, 2, 20
    critic_1_loss, critic_2_loss, policy_loss = 0, 0, 0
    generator_accuracy, expert_accuracy, dis_loss_gen, dis_loss_exp = 0, 0, 0, 0
    csv_path = "data/{}/{}_seed[{}].csv".format(args.env_name, args.env_name, args.seed)
    while True:
        # sampling generator data set from environment, type= numpy array, device:
        if iters % sample_interval == 0:
            with timed("sampling from environment segment len:{}".format(args.segment_len)):
                gene_data = segment_generator.__next__()
                # seg["state"]:arrary, seg["action"]:array, seg["entropy"]:tensor,shape:segment_len*action_space
        else:
            gene_data = segment_generator.__next__()
        # print("args.is_normal", args.is_normal)
        # print("generator_replay_batch_size", args.generator_replay_batch_size)
        # print("gen_replay_size", args.gen_replay_size)
        # sampling expert data set from expert demonstrations, type: torch tensor, device:cpu
        expert_state, expert_action, expert_next_state = expert_dataset.get_next_expert_batch(args.expert_batch)

        # a = np.concatenate((gene_data["state"], gene_data["action"]), axis=1)
        # b = np.concatenate((expert_state, expert_action), axis=1)
        # In total_data: 0 - (args.segment_len-1) row is generated data,
        # args.segment_len - (args.segment_len+args.expert_batch-1) row is expert data
        if args.is_normal:
            gene_state_memory = (gene_data["state"] - state_mean) / state_std
            gene_action_memory = (gene_data["action"] - action_mean) / action_std
            gene_next_state_memory = (gene_data["next_state"] - state_mean) / state_std
            expert_state_memory = (expert_state - state_mean) / state_std
            expert_action_memory = (expert_action - action_mean) / action_std
            expert_next_state_memory = (expert_next_state - state_mean) / state_std
        else:
            gene_state_memory = gene_data["state"]
            gene_action_memory = gene_data["action"]
            gene_next_state_memory = gene_data["next_state"]
            expert_state_memory = expert_state
            expert_action_memory = expert_action
            expert_next_state_memory = expert_next_state

        # gene_total_data_normal = np.concatenate((gen_state_normal, gen_action_normal), axis=1)

        # exp_state_normal = (expert_state - state_mean) / state_std
        # exp_action_normal = (expert_action - action_mean) / action_std
        # exp_next_state_normal = (expert_next_state - state_mean) / state_std
        # expert_total_data_normal = np.concatenate((exp_state_normal, exp_action_normal), axis=1)

        # total_data_normal = np.vstack((gene_total_data_normal, expert_total_data_normal))
        total_reward = discriminator.discriminator(torch.from_numpy(
            np.vstack((np.concatenate((gene_data["state"], gene_data["action"]), axis=1),
                       np.concatenate((expert_state, expert_action), axis=1))))
                                                   .to(torch.float32).to(device))
        # expert_accuracy = np.array(total_reward[args.segment_len:].cpu() > 0.5).mean()
        # generator_accuracy = np.array(total_reward[:args.segment_len].cpu() < 0.5).mean()
        # gene_reward = total_reward[:args.segment_len]
        # # # Creating a mix_index to random mix expert data and generate data
        # # # In the original total_data, 0-99 row is generated data, 100-199 row is expert data
        # # mix_index = np.arange(2*args.segment_len)
        # # np.random.shuffle(mix_index)
        # # # the data what is mixed, shape:(2*args.segment_len, state_space+action_space)
        # # total_data = total_data[mix_index, :]
        # # Compute reward from discriminator
        # # discriminator_reward = discriminator.discriminator(torch.from_numpy(total_data).to(torch.float32).to(device))
        # # Computer accuracy of discriminator
        # # generator_accuracy = np.array(discriminator_reward[:args.segment_len - 1].cpu() < 0.5).mean()
        # # expert_accuracy = np.array(discriminator_reward[args.segment_len:].cpu() > 0.5).mean()
        #
        # # if expert_accuracy < 0.9:
        if iters % args.dis_interval == 0:
            for i in range(args.d_steps):
                # Compute reward from discriminator
                dis_loss_exp = discriminator.update_discriminator(dsa=total_reward, segment_len=args.segment_len,
                                                                  exp_current_len=args.dis_upd_exp_len,
                                                                  gen_current_len=args.dis_upd_gen_len)
                generator_accuracy = np.array(total_reward[:args.segment_len].cpu() < 0.5).mean()
                expert_accuracy = np.array(total_reward[args.segment_len:].cpu() > 0.5).mean()

        # Push generated data and expert data to replay buffer
        for i in range(args.segment_len):
            if generator_accuracy > 0.6 or generator_accuracy < 0.4:
                generator_memory.push(gene_state_memory[i], gene_action_memory[i],
                                      np.array([0]) if total_reward[:args.segment_len][i] < 0.2
                                      else np.array([-torch.log(1 - total_reward[:args.segment_len][i]
                                                                + args.reward_baseline).item()]),
                                      gene_next_state_memory[i], gene_data["mask"][i])
            else:
                generator_memory.push(gene_state_memory[i], gene_action_memory[i],
                                      np.array([-torch.log(torch.tensor(args.reward_baseline)).item()]),
                                      gene_next_state_memory[i], gene_data["mask"][i])
            # -torch.log(1 - gene_reward[i] + 1e-1).item()
        for i in range(args.expert_batch):
            generator_memory.push(expert_state_memory[i], expert_action_memory[i],
                                  np.array([-torch.log(torch.tensor(args.reward_baseline)).item()]),
                                  expert_next_state_memory[i], np.array([1]))
        if len(generator_memory) < args.generator_replay_batch_size:
            continue

        for _ in range(args.g_steps):
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                generator.update_parameters(generator_memory, args.generator_replay_batch_size, 0,
                                            gene_entropy=torch.mean(gene_data["entropy"]).item())

        if iters % print_interval == 0:
            # Test generator
            avg_reward = 0.
            episodes = 4
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    # state_normal = (state - state_mean) / state_std
                    action, _ = generator.select_action(state, evaluate=True)

                    # action_normal = (action - action_mean) / action_std
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            avg_reward1 = avg_reward
            # print
            print("iters:{}, total_steps:{}, episode_return:{}, verified_episode_return:{}, "
                  "generator_accuracy:{}, expert_accuracy:{}"
                  .format(iters, gene_data["total_steps"], round(avg_reward, 4),
                          round(gene_data["episode_return"], 3), round(generator_accuracy, 3),
                          round(expert_accuracy, 3)))
            print("scrip_num:{}, disc_loss_gen:{}, dis_loss_exp:{}, critic_1_loss:{}, critic_2_loss:{}, policy_loss:{}"
                  .format(args.scrip_num, round(dis_loss_gen, 4), round(dis_loss_exp, 4), round(critic_1_loss, 4),
                          round(critic_2_loss, 4), round(policy_loss, 4)))

            # writer data to csv
            # print(color_text.cyan("--" * 10, ))
            # data = [round(avg_reward, 4, 3), iters]
            # with open(csv_path, "a+", newline='') as f:
            #     print_text = "-----------{} added a new line!------------".format(csv_path)
            #     print(color_text.cyan(print_text))
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow(data)
            # verifying the generator performence
        if iters % writer_interval == 0:
            # Writing data to tensorboard
            writer.add_scalar("verified_episode_return", round(avg_reward1, 4), gene_data["total_steps"])
            writer.add_scalar("policy_loss", round(policy_loss, 4), iters)
            writer.add_scalar("critic_1_loss", round(critic_1_loss, 4), iters)
            writer.add_scalar("critic_2_loss", round(critic_2_loss, 4), iters)
            writer.add_scalar("expert_accuracy", round(expert_accuracy, 4), iters)
            writer.add_scalar("generator_accuracy", round(generator_accuracy, 4), iters)
            writer.add_scalar("scrip_num", args.scrip_num, iters)

        iters += 1
        if gene_data["total_steps"] > args.num_steps:
            break
    print("done!")
    # 先更新 discriminator，然后异步更新 generator，
    env.close()
    writer.close()


def main(envs):
    # Args
    global_args = parser.get_global_passer()
    args = get_env_parser(envs, global_args)
    # from env_parser.AntParser import get_parser
    # args = get_parser()
    # Cuda of CPU
    if args.cuda and torch.cuda.is_available():
        with torch.cuda.device(args.GPU_num):
            train_gail(args)
    else:
        train_gail(args)


if __name__ == '__main__':
    environments = ["Ant-v2", "Hopper-v2", "HalfCheetah-v2", "walker2d-v2",
                    "InvertedPendulum-v2", "InvertedDoublePendulum-v2"]
    main(envs=environments)
