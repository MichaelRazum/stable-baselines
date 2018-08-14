import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.input import observation_input


def nature_cnn(unscaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param unscaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class ActorCriticPolicy(object):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False):
        """
        Policy object that implements actor critic

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param n_lstm: (int) The number of LSTM cells (for reccurent policies)
        :param n_stack: (int) The number of frames stacked (None for no stacking)
        :param reuse: (bool) If the policy is reusable or not
        """
        self.n_env = n_env
        self.n_steps = n_steps
        self.obs_ph, self.processed_x = observation_input(ob_space, n_batch, n_stack=n_stack)
        self.masks_ph = tf.placeholder(tf.float32, [n_batch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2])  # states
        self.pdtype = make_proba_dist_type(ac_space)
        self.sess = sess
        self.reuse = reuse

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class LstmPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 layer_norm=False, _type="cnn", **kwargs):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse)
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.processed_x, **kwargs)
            else:
                activ = tf.tanh
                extracted_features = tf.layers.flatten(self.processed_x)
                extracted_features = activ(linear(extracted_features, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                extracted_features = activ(linear(extracted_features, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.action = self.proba_distribution.sample()
        self.neglogp = self.proba_distribution.neglogp(self.action)
        self.policy_proba = tf.nn.softmax(self.policy)
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self.value_fn = value_fn
        self._value = value_fn[:, 0]

    def step(self, obs, state=None, mask=None):
        return self.sess.run([self.action, self._value, self.snew, self.neglogp],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 _type="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack,
                                                reuse)
        if "layer_norm" in kwargs:
            kwargs.pop("layer_norm")  # ignore lstm keyword
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.processed_x, **kwargs)
                value_fn = linear(extracted_features, 'vf', 1)
                pi_latent = extracted_features
                vf_latent = extracted_features
            else:
                activ = tf.tanh
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h1 = activ(linear(processed_x, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                pi_h2 = activ(linear(pi_h1, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
                vf_h1 = activ(linear(processed_x, 'vf_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                vf_h2 = activ(linear(vf_h1, 'vf_fc2', n_hidden=64, init_scale=np.sqrt(2)))
                value_fn = linear(vf_h2, 'vf', 1)
                pi_latent = pi_h2
                vf_latent = vf_h2

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.action = self.proba_distribution.sample()
        self.neglogp = self.proba_distribution.neglogp(self.action)
        self.policy_proba = tf.nn.softmax(self.policy)
        self.initial_state = None
        self.value_fn = value_fn
        self._value = value_fn[:, 0]

    def step(self, obs, state=None, mask=None):
        action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                        _type="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                            layer_norm=False, _type="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                              layer_norm=True, _type="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                        _type="mlp", **_kwargs)


class MlpLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                            layer_norm=False, _type="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, n_stack=None, reuse=False,
                 **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, n_stack, reuse,
                                              layer_norm=True, _type="mlp", **_kwargs)