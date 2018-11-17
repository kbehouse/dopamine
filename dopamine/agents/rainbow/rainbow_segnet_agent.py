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

# from dopamine.agents.rainbow.VGGSegnet import VGGSegnet
from dopamine.agents.rainbow.Encoder_Decoder import build_encoder_decoder
slim = tf.contrib.slim


# TWO_IMG_OBSERVATION_SHAPE = (1, 84, 84, 3)
STATE_W_H = 128
dqn_agent.OBSERVATION_SHAPE = (STATE_W_H, STATE_W_H, 3) 
dqn_agent.STACK_SIZE = 2

@gin.configurable
class RainbowSegNetAgent(RainbowAgent):
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
               obj_class = 2):

    print('--------in RainbowRGBAgent------')
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
    self.obj_class = obj_class
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

    
    # K.set_session(sess)

    with tf.device(tf_device):
      state_shape = [1, dqn_agent.OBSERVATION_SHAPE [0], dqn_agent.OBSERVATION_SHAPE[1], dqn_agent.OBSERVATION_SHAPE[2] * dqn_agent.STACK_SIZE]
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
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
    state: `tf.Tensor`, contains the agent's current state.

    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    
    print(' --------in RainbowSegNetAgent network_template---------')
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    s = state[:,:,:,-3:]
    print('s ->', s.shape)
    s = tf.cast(s, tf.float32)


    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32) # (3)
    mean = tf.reshape(mean, [1, 1, 3])
    s = s - mean

    # m = VGGSegnet(s , self.obj_class , input_height=STATE_W_H, input_width=STATE_W_H   )
    segnet_ouput_tensor = build_encoder_decoder(s, 2)
    # segnet_ouput_tensor = m.output
    # segnet_ouput_tensor = net 
    print('segnet_ouput_tensor =', segnet_ouput_tensor)
    print('segnet_ouput_tensor.shape =', segnet_ouput_tensor.shape)
    # segnet_ouput_tensor = tf.argmax(segnet_ouput_tensor, axis= -1)
    # net = tf.cast(state, tf.float32)
    # net = tf.div(net, float(self.obj_class))
    # print('segnet_ouput_tensor.shape =', segnet_ouput_tensor.shape)
    '''
    net = slim.conv2d( segnet_ouput_tensor, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    net = slim.conv2d( net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    net = slim.conv2d( net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    # print(' conv2d 64, [3, 3], stride=1 , net -> ', net

    # net = tf.contrib.layers.spatial_softmax(net)
    net = slim.flatten(net)
    '''
    net = slim.flatten(segnet_ouput_tensor)
    # print('self.output_layer -> ', output_layer)
    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    # print(' 512 , net -> ', net)
    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    print(' fully_connected , net -> ', net)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    print('probabilities -> ', probabilities)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    print('q_values -> ', q_values)
    return self._get_network_type()(q_values, logits, probabilities, segnet_ouput_tensor)


  def _network_template_old(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
    state: `tf.Tensor`, contains the agent's current state.

    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    
    print(' --------in RainbowRGBAgent network_template---------')
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    # print('!!!!!!!1 tf.float32 , net -> ', net)
    net = tf.div(net, 255.)
    # print(' div 255 , net -> ', net)
    net = slim.conv2d( net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    # print(' conv2d 32, [8,8], stride=4 , net -> ', net)
    net = slim.conv2d( net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    # print(' conv2d 64, [4, 4], stride=2 , net -> ', net)
    net = slim.conv2d( net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    # print(' conv2d 64, [3, 3], stride=1 , net -> ', net)


    net = tf.contrib.layers.spatial_softmax
    # self.output_layer = net
    output_layer = net
    print('self.output_layer -> ', output_layer)
    net = slim.flatten(net)
    print(' flatten , net -> ', net)
    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    # print(' 512 , net -> ', net)
    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    print(' fully_connected , net -> ', net)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    print('probabilities -> ', probabilities)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    print('q_values -> ', q_values)
    return self._get_network_type()(q_values, logits, probabilities, output_layer)

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
    self.state = np.roll(self.state, -3, axis=3)
    self.state[0, :, :, -3:] = self._observation

    # print('observation.shape = ', np.shape(observation)) # observation.shape =  (256, 256, 3)
    # print('np.shape(self.state) = ', np.shape(self.state)) # self.state =  (1, 256, 256, 12)
    # print('self._observation.shape = ', np.shape(self._observation)) # self._observation.shape =  (256, 256, 3)

  def get_output_layer(self):
    return self._sess.run(self._net_outputs.know_output_layer, {self.state_ph: self.state})

  def get_probabilities(self):
    return self._sess.run(self._net_outputs.probabilities, {self.state_ph: self.state})

  def get_q_values(self):
    return self._sess.run(self._net_outputs.q_values, {self.state_ph: self.state})


