import tensorflow as tf
import numpy as np
import glfw
from time import time
from types import MethodType
from contextlib import contextmanager


def env_wrapper(env):
    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 1920, 1080
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    env.unwrapped.close = MethodType(close, env.unwrapped)
    env.unwrapped.render = MethodType(render, env.unwrapped)
    return env


class DiagGaussianPd:
    def __init__(self, mean, logstd):
        # self.flat = flat
        # mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    # def flatparam(self):
    #     return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=tf.float32)

    def logp(self, x):
        return - self.neglogp(x)


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-2):
        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        self.newcount = tf.placeholder(shape=(), dtype=tf.float64, name='count')
        self.update_list = [tf.assign_add(self._sum, self.newsum),
                            tf.assign_add(self._sumsq, self.newsumsq),
                            tf.assign_add(self._count, self.newcount)]

    def update(self, x):
        x = x.astype('float64')
        feed_dict = {self.newsum: x.sum(axis=0),
                     self.newsumsq: np.square(x).sum(axis=0),
                     self.newcount: len(x)}
        tf.get_default_session().run(self.update_list, feed_dict)

RANK = 0
START_TIME = time()


@contextmanager
def logger(info):
    global RANK
    if RANK == 0:
        print("%.2f: start %s" % (time() - START_TIME, info))
    elif RANK == 1:
        print("=" * 16)
        print("%.2f: start %s" % (time() - START_TIME, info))
    elif RANK == 2:
        print("\t" + "-" * 12)
        print("\t%.2f: start %s" % (time() - START_TIME, info))
    else:
        print("\t" * (RANK - 1) + "%.2f: start %s" % (time() - START_TIME, info))
    RANK += 1
    yield
    if RANK == 1:
        print("%.2f: end %s" % (time() - START_TIME, info))
    elif RANK == 2:
        print("%.2f: end %s" % (time() - START_TIME, info))
        print("=" * 16)
    elif RANK == 3:
        print("\t%.2f: end %s" % (time() - START_TIME, info))
        print("\t" + "-" * 12)
    else:
        print("\t" * (RANK - 2) + "%.2f: end %s" % (time() - START_TIME, info))
    RANK -= 1
