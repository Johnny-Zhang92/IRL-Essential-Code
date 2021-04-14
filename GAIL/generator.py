import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Generator:
    def __init__(self, pi, env: gym.Env, reward_giver, n_step, record_path=None):
        self.pi = pi    # policy
        self.env = env  # environment for simulation
        self.reward_giver = reward_giver
        self.n_step = n_step
        self.path = record_path
        pass

    def sample_trajectory(self, stochastic=True, display=False, record=False):
        # Initialize state variables
        t = 0
        ac = self.env.action_space.sample()
        new = True      # whether a new episode begins
        reward = 0.0    # reward predicted by value function
        true_reward = 0.0   # reward calculated according to all rewards
        vpred = 0.0
        st = np.zeros(self.reward_giver.st_shape, np.float32)
        ob = self.env.reset()

        if record:
            rec = VideoRecorder(self.env, path=self.path)

        # these are designed for multi episodes in one sample
        # cur_ep_ret = 0
        # cur_ep_len = 0
        # cur_ep_true_ret = 0
        # ep_true_rets = []
        # ep_rets = []
        # ep_lens = []

        # Initialize history arrays
        obs = np.array([ob for _ in range(self.n_step)])
        acs = np.array([ac for _ in range(self.n_step)])
        pre_acs = acs.copy()  # deep copy
        sts = np.ndarray((self.n_step,)+self.reward_giver.st_shape, np.float32)
        true_rewards = np.zeros(self.n_step, np.float32)
        rewards = np.zeros(self.n_step, np.float32)
        vpreds = np.zeros(self.n_step, np.float32)
        news = np.zeros(self.n_step, np.int32)

        for i in range(self.n_step):
            # record the previous data
            pre_ac = ac
            obs[i] = ob
            pre_acs[i] = pre_ac
            news[i] = new
            # perform policy and record
            ac, vpred = self.pi.act(stochastic, ob)
            acs[i] = ac
            vpreds[i] = vpred
            # evaluate values and record
            if self.reward_giver is not None:
                reward, st = self.reward_giver.get_reward(ob, st)
                sts[i] = st
            else:
                reward = 0
            rewards[i] = reward
            # take action and record true reward
            ob, true_reward, new, _ = self.env.step(ac)
            if record:
                rec.capture_frame()
            elif display:
                self.env.render()
            true_rewards[i] = true_reward
            if new:
                ob = self.env.reset()
                st = np.zeros(self.reward_giver.st_shape, np.float32)

        if display:
            self.env.close()
        return {"ob": obs,
                "reward": rewards,
                "vpred": vpreds,
                "ac": acs,
                "pre_ac": pre_acs,
                "new": news,
                "nextvpred": vpred * (1 - new),
                "st": sts}

    @staticmethod
    def process_trajectory(traj, gamma, lam):
        new = np.append(traj["new"],
                        0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(traj["vpred"], traj["nextvpred"])
        T = len(traj["reward"])
        traj["adv"] = gaelam = np.empty(T, 'float32')
        rewards = traj["reward"]
        # discounted rewards:
        # A_pre + v_pre = gamma*(A*lambda + v) + r_pre
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        traj["tdlamret"] = traj["adv"] + traj["vpred"]  # target of value function
        return traj
