from gym.envs.mujoco import mujoco_env
import numpy as np


class UR5(mujoco_env.MujocoEnv):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "/File/LAB/project/envs/assets/UR5gripper.xml", 2)

    def step(self, a):
        # vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = 0
        reward_ctrl = 0
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2]
        ])


if __name__ == '__main__':
    env = UR5()
    print(env.action_space)
    while True:
        env.render()
