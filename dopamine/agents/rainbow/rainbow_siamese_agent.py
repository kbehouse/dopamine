"""
Add Siamese Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
import numpy as np
import tensorflow as tf

import gin.tf
import random
slim = tf.contrib.slim


TWO_IMG_OBSERVATION_SHAPE = (2, 84, 84)

@gin.configurable
class RainbowSiameseAgent(RainbowAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):

    with tf.device(tf_device):
      state_shape = [1, dqn_agent.OBSERVATION_SHAPE, dqn_agent.OBSERVATION_SHAPE, dqn_agent.STACK_SIZE]
      self.state_second = np.zeros(state_shape)
      self.state_second_ph = tf.placeholder(tf.uint8, state_shape, name='state_second_ph')
    super(RainbowSiameseAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    first_state = state[0]
    second_state = state[1]
    # first network
    first_net = tf.cast(first_state, tf.float32)
    first_net = tf.div(first_net, 255.)
    first_net = slim.conv2d(
        first_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    first_net = slim.conv2d(
        first_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    first_net = slim.conv2d(
        first_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    first_net = slim.flatten(first_net)
    first_net = slim.fully_connected(
        first_net, 512, weights_initializer=weights_initializer)

    # second network
    second_net = tf.cast(second_state, tf.float32)
    second_net = tf.div(second_net, 255.)
    second_net = slim.conv2d(
        second_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    second_net = slim.conv2d(
        second_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    second_net = slim.conv2d(
        second_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    second_net = slim.flatten(second_net)
    second_net = slim.fully_connected(
        second_net, 512, weights_initializer=weights_initializer)

    net = tf.concat([first_net, second_net], axis=1)

    print('first_net', first_net)
    print('second_net', second_net)
    print('net', net)

    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)

    print('net', net)

    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(q_values, logits, probabilities)
  
  '''
  def _network_template(self, state, second_state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    # first network
    first_net = tf.cast(state, tf.float32)
    first_net = tf.div(first_net, 255.)
    first_net = slim.conv2d(
        first_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    first_net = slim.conv2d(
        first_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    first_net = slim.conv2d(
        first_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    first_net = slim.flatten(first_net)
    first_net = slim.fully_connected(
        first_net, 512, weights_initializer=weights_initializer)

    # second network
    second_net = tf.cast(second_state, tf.float32)
    second_net = tf.div(second_net, 255.)
    second_net = slim.conv2d(
        second_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    second_net = slim.conv2d(
        second_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    second_net = slim.conv2d(
        second_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    second_net = slim.flatten(second_net)
    second_net = slim.fully_connected(
        second_net, 512, weights_initializer=weights_initializer)

    net = tf.concat([first_net, second_net], axis=1)

    print('first_net', first_net)
    print('second_net', second_net)
    print('net', net)

    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)

    print('net', net)

    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(q_values, logits, probabilities)
  
  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    print('in RainbowSiameseAgent _select_action')
    epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state.
      return self._sess.run(self._q_argmax, {self.state_ph: self.state,  \
                                             self.state_second_ph: self.state_second})

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph, self.state_second_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)
  '''