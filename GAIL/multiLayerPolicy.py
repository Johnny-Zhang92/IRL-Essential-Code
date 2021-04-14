import tensorflow as tf
import numpy as np
import gym

from util import (RunningMeanStd, DiagGaussianPd)


class MultiLayerPolicy:
    def __init__(self, name, ob, ac_shape, hid_size=128, num_hid_layers=3, reuse=False):
        with tf.variable_scope(name, reuse):
            self.scope = tf.get_variable_scope().name
            self.build_net(ob, ac_shape, hid_size, num_hid_layers)

    def build_net(self, ob, ac_shape, hid_size, num_hid_layers):
        self.ob = ob
        self.ob_shape = ob.shape.as_list()[1:]

        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(ob.shape.as_list()[1:])

        # normalized observation
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # net to fit value function
        net = obz
        for i in range(num_hid_layers):
            net = tf.layers.dense(inputs=net,
                                  units=hid_size,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                  name="vffc%i" % (i + 1))
        self.vpred = tf.layers.dense(inputs=net,
                                     units=1,
                                     activation=None,
                                     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                     name="vffinal")
        # train value function
        self.vreal = tf.placeholder(dtype=tf.float32, shape=(None,), name="vreal")
        vloss = tf.reduce_mean(tf.square(self.vreal-self.vpred))
        valueFunctionVars = [v for v in self.get_trainable_variables() if v.name.startswith("%s/vff" % self.scope)]
        self.vadam = tf.train.AdamOptimizer().minimize(vloss, var_list=valueFunctionVars)

        # net to predict mean and standard deviation of action
        net = obz
        for i in range(num_hid_layers):
            net = tf.layers.dense(inputs=net,
                                  units=hid_size,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                  name="polc%i" % (i + 1))
        mean = tf.layers.dense(inputs=net,
                               units=ac_shape[0],
                               activation=None,
                               kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        logstd = mean * 0.0 + tf.get_variable(name="logstd",
                                              shape=[1, ac_shape[0]],
                                              initializer=tf.zeros_initializer(),
                                              dtype=tf.float32)  # std not related to observation

        # action is normally distributed
        self.pd = DiagGaussianPd(mean, logstd)
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic")
        self.action = tf.cond(self.stochastic,
                              lambda: self.pd.sample(),
                              lambda: self.pd.mode())

    def act(self, stochastic, ob):
        action, vpred = tf.get_default_session().run([self.action, self.vpred],
                                                     {self.ob: ob[None], self.stochastic: stochastic})
        return action[0], vpred[0]

    def train_value_function(self, obs, vreals):
        self.ob_rms.update(obs)
        tf.get_default_session().run([self.vadam],
                                     {self.ob: obs, self.vreal: vreals})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


def main():
    ob = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="ob")
    pi = MultiLayerPolicy("pi", ob, (2,))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(pi.act(True, np.array([1.0, 1.0])))
        pass


if __name__ == '__main__':
    main()
