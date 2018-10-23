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
import math
from dopamine.replay_memory import circular_replay_buffer
slim = tf.contrib.slim


STATE_W_H = 84
dqn_agent.OBSERVATION_SHAPE = (2, STATE_W_H, STATE_W_H, 3) 
dqn_agent.STACK_SIZE = 4

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
               summary_writing_frequency=500, 
               hsv_color = False):



    print('--------in RainbowRGBAgent------')
    print('min_replay_history = ', min_replay_history)
    # tf.logging.info('Creating %s agent with the following parameters:',
    #                 self.__class__.__name__)
    # tf.logging.info('\t gamma: %f', gamma)
    # tf.logging.info('\t update_horizon: %f', update_horizon)
    # tf.logging.info('\t min_replay_history: %d', min_replay_history)
    # tf.logging.info('\t update_period: %d', update_period)
    # tf.logging.info('\t target_update_period: %d', target_update_period)
    # tf.logging.info('\t epsilon_train: %f', epsilon_train)
    # tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
    # tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    # tf.logging.info('\t tf_device: %s', tf_device)
    # tf.logging.info('\t use_staging: %s', use_staging)
    # tf.logging.info('\t optimizer: %s', optimizer)
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.optimizer = optimizer
    

    self.num_actions = num_actions
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = False
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency

    self.hsv_color = hsv_color
    with tf.device(tf_device):
      state_shape = [1, dqn_agent.OBSERVATION_SHAPE [0], dqn_agent.OBSERVATION_SHAPE[1],dqn_agent.OBSERVATION_SHAPE[2], dqn_agent.OBSERVATION_SHAPE[3] * dqn_agent.STACK_SIZE]
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(tf.uint8, state_shape, name='state_ph')

      self._replay = self._build_replay_buffer(use_staging)

      self._build_networks()

      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

      print('self.state_ph= ', self.state_ph)

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._saver = tf.train.Saver(max_to_keep=3)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None
    

  def _network_template(self, state):
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    # first_state  = state[0]
    # first_state  = tf.expand_dims(first_state, 0)
    # second_state = state[1]
    # second_state = tf.expand_dims(second_state, 0)
    first_state  = state[:,0,:,:,:]
    second_state = state[:,1,:,:,:3]
    print('state shape = ', state.shape)
    print('first_state shape = ', first_state.shape)
    print('second_state shape = ', second_state.shape)
    # first network
    first_net = tf.cast(first_state, tf.float32)
    if not self.hsv_color:
        first_net = tf.div(first_net, 255.)
    # if self.hsv_color:
    #     print('Network uses HSV Color!!!')
    #     first_net_rgb_float = tf.image.convert_image_dtype(first_net, tf.float32)
    #     first_net = tf.image.rgb_to_hsv(first_net_rgb_float)  
    # else: # rgb color
    #     first_net = tf.div(first_net, 255.)
    first_net = slim.conv2d(first_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    first_net = slim.conv2d(first_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    first_net = slim.conv2d(first_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    first_net = slim.flatten(first_net)
    first_net = slim.fully_connected(first_net, 1024, weights_initializer=weights_initializer)

    # second network
    second_net = tf.cast(second_state, tf.float32)
    if not self.hsv_color:
        second_net = tf.div(second_net, 255.)
    # if self.hsv_color:
    #     second_net_rgb_float = tf.image.convert_image_dtype(second_net, tf.float32)
    #     second_net = tf.image.rgb_to_hsv(second_net_rgb_float)  
    # else: # rgb color
    #     second_net = tf.div(second_net, 255.)
    second_net = slim.conv2d(second_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    second_net = slim.conv2d(second_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    second_net = slim.conv2d(second_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    second_net = slim.flatten(second_net)
    second_net = slim.fully_connected(second_net, 1024, weights_initializer=weights_initializer)

    # net = tf.concat([first_net, second_net], axis=1)
    net = tf.subtract(first_net, second_net)
    print('first_net', first_net)
    print('second_net', second_net)
    print('net', net)

    net = slim.fully_connected(net, 512, weights_initializer=weights_initializer)
    net = slim.fully_connected(net, 256, weights_initializer=weights_initializer)

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

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
    observation: numpy array, an observation from the environment.
    """
    # print('in rainbow_rgb_agent _record_observation')
    
    # Set current observation. Represents an 84 x 84 x 1 image frame.
    self._observation = observation
    # Swap out the oldest frame with the current frame.
    self.state = np.roll(self.state, -3, axis=4)
    self.state[0, :, :, :, -3:] = self._observation

    # print('observation.shape = ', np.shape(observation)) # observation.shape =  (2, 256, 256, 3)
    # print('np.shape(self.state) = ', np.shape(self.state)) # self.state =  (1, 2, 256, 256, 12)
    # print('self._observation.shape = ', np.shape(self._observation)) # self._observation.shape =  (256, 256, 3)

    # print('observation.shape = ', np.shape(observation)) # observation.shape =  (256, 256, 3)
    # print('np.shape(self.state) = ', np.shape(self.state)) # self.state =  (1, 256, 256, 12)
    # print('self._observation.shape = ', np.shape(self._observation)) # self._observation.shape =  (256, 256, 3)
