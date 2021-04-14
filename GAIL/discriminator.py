import tensorflow as tf
import numpy as np
from util import RunningMeanStd
import os


def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits + tf.nn.softplus(-logits)
    return ent


class Discriminator:
    def __init__(self, name, ob_shape, st_shape=(4,), hid_size=128, ent_coff=0.001, ob_slice=None):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self.st_shape = st_shape
            self.build_net(ob_shape, st_shape, hid_size, ent_coff)
            if ob_slice is not None:
                assert len(ob_slice) == ob_shape[0]
                self.ob_slice = ob_slice
            else:
                self.ob_slice = range(ob_shape)
            if not os.path.exists("./log/discriminator"):
                os.mkdir("./log/discriminator")
            self.writer = tf.summary.FileWriter("./log/discriminator")

    def build_net(self, ob_shape, st_shape, hid_size, ent_coeff):
        # build placeholders
        self.generator_obs = tf.placeholder(tf.float32, (None,) + ob_shape, name="generator_observations")
        self.generator_sts = tf.placeholder(tf.float32, (None,) + st_shape, name="generator_states")
        self.expert_obs = tf.placeholder(tf.float32, (None,) + ob_shape, name="expert_observations")
        self.expert_sts = tf.placeholder(tf.float32, (None,) + st_shape, name="expert_states")

        # normalize observation
        with tf.variable_scope("obfilter"):
            self.obs_rms = RunningMeanStd(shape=ob_shape)

        # network to judge generator
        net = (self.generator_obs-self.obs_rms.mean)/self.obs_rms.std
        with tf.variable_scope("main_net", reuse=False):
            net, self.generator_next_sts = tf.nn.rnn_cell.BasicRNNCell(num_units=st_shape[0])(net, self.generator_sts)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            generator_logits = tf.layers.dense(inputs=net, units=1, activation=tf.identity)

        # network to judge expert
        net = (self.expert_obs-self.obs_rms.mean)/self.obs_rms.std
        with tf.variable_scope("main_net", reuse=True):
            net, self.expert_next_sts = tf.nn.rnn_cell.BasicRNNCell(num_units=st_shape[0])(net, self.expert_sts)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            net = tf.layers.dense(inputs=net, units=hid_size, activation=tf.nn.tanh)
            expert_logits = tf.layers.dense(inputs=net, units=1, activation=tf.identity)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        self.merged = tf.summary.merge([tf.summary.scalar("Expert accuracy", expert_acc),
                                        tf.summary.scalar("Generator accuracy", generator_acc)])

        # loss for the two networks respectively
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -ent_coeff * entropy

        # reward and optimizer
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.adam = tf.train.AdamOptimizer().minimize(loss=self.total_loss,
                                                      var_list=self.get_trainable_variable())

    def get_trainable_variable(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, sts):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(sts.shape) == 1:
            sts = np.expand_dims(sts, 0)
        feed_dict = {self.generator_obs: obs[:, self.ob_slice], self.generator_sts: sts}
        return tf.get_default_session().run([self.reward_op, self.generator_next_sts], feed_dict)

    def train(self, generator_obs, generator_sts, expert_obs, expert_sts):
        self.obs_rms.update(np.concatenate([generator_obs[:, self.ob_slice], expert_obs[:, self.ob_slice]], 0))
        _, summary = tf.get_default_session().run([self.adam, self.merged],
                                                  {self.generator_obs: generator_obs[:, self.ob_slice],
                                                   self.generator_sts: generator_sts,
                                                   self.expert_obs: expert_obs[:, self.ob_slice],
                                                   self.expert_sts: expert_sts})
        try:
            self.summary_step += 1
        except AttributeError:
            self.summary_step = 0
        finally:
            self.writer.add_summary(summary, self.summary_step)


def main():
    d = Discriminator("discriminator", (11,), (2,), 1024)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        pass


if __name__ == '__main__':
    main()
