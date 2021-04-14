from gym.envs.mujoco import mujoco_env
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import glfw
import mujoco_py


class Reacher3Link(mujoco_env.MujocoEnv):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "/File/LAB/project/envs/assets/reacher_3link.xml", 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, None, done, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if 0.01 < np.linalg.norm(self.goal) < 0.21:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:3]
        return np.concatenate([
            self.sim.data.qpos.flat[3:],
            self.get_body_com("fingertip")[:2],
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:3]
        ])

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        if mode == 'rgb_array':
            self.viewer.render()
            # window size used for old mujoco-py:
            width, height = 1920, 1080
            data = self.viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self.viewer.render()


class Reacher2Link(mujoco_env.MujocoEnv):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "/File/LAB/project/envs/assets/reacher_2link.xml", 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, None, done, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if 0.01 < np.linalg.norm(self.goal) < 0.21:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.get_body_com("fingertip")[:2],
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:2]
        ])

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        if mode == 'rgb_array':
            self.viewer.render()
            # window size used for old mujoco-py:
            width, height = 1920, 1080
            data = self.viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self.viewer.render()


def reacher(n_links=2):
    if n_links==2:
        return TimeLimit(Reacher2Link(),
                         max_episode_steps=50,
                         max_episode_seconds=None)
    elif n_links==3:
        return TimeLimit(Reacher3Link(),
                         max_episode_steps=50,
                         max_episode_seconds=None)


if __name__ == '__main__':
    env = reacher(3)
    env.reset()
    for _ in range(50):
        env.render()
        ob, _, done, _ = env.step(np.array([1e-3, 1e-3, 1e-3]))
        print(done)
    env.close()
