import os.path as osp

import numpy as np
import gym
import torch

from sac import SAC
from gather_expert_demonstration.model_passer import get_passer


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, render, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    obs1 = []
    rews = []
    news = []  # env information:done
    acs = []

    while True:
        ac = pi.select_action(ob, evaluate=True)  # select action without noise
        obs.append(ob)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
        news.append(new)
        rews.append(rew)
        obs1.append(ob)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            # obs1.append(ob)  # next state is current state
            break
        t += 1

    obs1 = np.array(obs1)
    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)  # env: done
    acs = np.array(acs)

    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "obs1": obs1}

    return traj


def runner(env, actor_path, critic_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False, args=None, render=True):
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    # initial network
    pi = SAC(ob_space.shape[0], ac_space, args)
    # pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    # U.initialize()

    # Prepare for rollouts load model
    # ----------------------------------------
    pi.load_model(actor_path=actor_path, critic_path=critic_path)
    # u.load_variables(load_model_path)

    obs_list = []
    obs1_list = []
    acs_list = []
    reward_list = []
    done_list = []
    episode_len_list = []
    episode_return_list = []
    current_traj_num = 0
    current_abandon_traj_num = 0
    while current_traj_num <= number_trajs:
        traj = traj_1_generator(pi, env, timesteps_per_batch, render, stochastic=stochastic_policy)
        if traj['ep_len'] < timesteps_per_batch:
            current_abandon_traj_num += 1
            print("abandon episode number:{}!, episode len:{}".format(current_abandon_traj_num, traj['ep_len']))
            continue
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs1, reward, done = traj['obs1'], traj['rew'], traj['new']

        # append multi dimension array
        obs_list.append(obs)
        obs1_list.append(obs1)
        acs_list.append(acs)
        episode_len_list.append(ep_len)
        episode_return_list.append(ep_ret)
        done_list.append(done)
        reward_list.append(reward)
        current_traj_num += 1
        if current_traj_num % 1 == 0:  # control the print frequency
            print("accept episode number:{}, len:{}, returns:{}".format(current_traj_num, ep_len, ep_ret))
        if current_traj_num >= number_trajs:
            break

    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        # Assemble the file name
        file_path = 'gather_expert_demonstration/expert_demonstration_data/model_guide/'
        file_name = 'stochastic' if stochastic_policy else 'deterministic' + '_SAC_' \
                                                           + env.spec.id + "(6000)" + '_johnny'
        path = osp.join(file_path, file_name)
        # Save the gathered data collections to the filesystem
        np.savez(path, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(episode_len_list), returns=np.array(episode_return_list),
                 done=np.array(done_list), reward=np.array(reward_list))
        print("saving demonstrations")
        print("  @: {}.npz".format(path))
        # save expert data for contrast algorithm sam
        # Assemble the file name
        file_path = 'gather_expert_demonstration/expert_demonstration_data/sam/'
        file_name = 'stochastic' if stochastic_policy else 'deterministic' + '_SAC_' \
                                                           + env.spec.id + "(6000)" + '_sam'
        path = osp.join(file_path, file_name)
        np.savez(path,
                 obs0=np.array(obs_list),
                 acs=np.array(acs_list),
                 env_rews=np.array(reward_list),
                 dones1=np.array(done_list),
                 obs1=np.array(obs1_list),
                 ep_lens=np.array(episode_len_list),
                 ep_env_rets=np.array(episode_return_list))
        print("saving demonstrations")
        print("  @: {}.npz".format(path))

    avg_len = sum(episode_len_list) / len(episode_len_list)
    avg_ret = sum(episode_return_list) / len(episode_return_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


if __name__ == '__main__':
    # get args
    args = get_passer()
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    # fixed seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # parameter
    timesteps_per_trajs = 1000  # 1000
    number_trajs = 1500  # 1500
    stochastic_policy = False  # use stochastic/deterministic policy to evaluate
    save_trajectory = True  # save the trajectories or not
    render = False  # True  # show mujoco image
    suffix = ''  # add additional information, default= ''
    actor_path = "../models/sac_actor_{}_{}".format(args.env_name, suffix)
    critic_path = "../models/sac_critic_{}_{}".format(args.env_name, suffix)
    # generate and save expert demonstration
    runner(env=env, actor_path=args.actor_path, critic_path=args.critic_path,
           timesteps_per_batch=timesteps_per_trajs, number_trajs=number_trajs,
           stochastic_policy=stochastic_policy, save=save_trajectory, args=args,
           render=render)
