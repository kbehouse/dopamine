"""
Add Siamese Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
from dopamine.replay_memory import prioritized_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
import numpy as np
import tensorflow as tf

import gin.tf
import random
import math
slim = tf.contrib.slim


# TWO_IMG_OBSERVATION_SHAPE = (1, 84, 84, 3)
STATE_W_H = 84
dqn_agent.OBSERVATION_SHAPE = (STATE_W_H, STATE_W_H, 3) 
dqn_agent.STACK_SIZE = 4

@gin.configurable
class RainbowRGBGripperAgent(RainbowAgent):
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

    with tf.device(tf_device):
      state_shape = [1, dqn_agent.OBSERVATION_SHAPE [0], dqn_agent.OBSERVATION_SHAPE[1], dqn_agent.OBSERVATION_SHAPE[2] * dqn_agent.STACK_SIZE]
      gripper_shape = [1, 1]
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(tf.uint8, state_shape, name='state_ph')
      self.gripper_ph = tf.placeholder(tf.uint8, gripper_shape, name='gripper_ph')

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

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
    use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
    `WrappedPrioritizedReplayBuffer` object.

    Raises:
    ValueError: if given an invalid replay scheme.
    """
    print('in RGBGripper rainbow   _build_replay_buffer')

    if self._replay_scheme not in ['uniform', 'prioritized']:
        raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=dqn_agent.OBSERVATION_SHAPE,
        stack_size=dqn_agent.STACK_SIZE,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        extra_storage_types=[ReplayElement('gripper', (), np.uint8)]) # , ReplayElement('next_gripper', (), np.uint8)



    '''
    in circular_replay_buffer
      def get_storage_signature(self):
    """Returns a default list of elements to be stored in this replay memory.

    Note - Derived classes may return a different signature.

    Returns:
    list of ReplayElements defining the type of the contents stored.
    """
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                    self._observation_dtype),
        ReplayElement('action', (), np.int32),
        ReplayElement('reward', (), np.float32),
        ReplayElement('terminal', (), np.uint8)
    ]

    for extra_replay_element in self._extra_storage_types:
    storage_elements.append(extra_replay_element)
    return storage_elements
    '''

  def _network_template(self, state, gripper):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
    state: `tf.Tensor`, contains the agent's current state.

    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    if len(gripper.shape)==1:
      gripper = tf.expand_dims(gripper, axis=1)

    # print(' --------in RainbowRGBAgent network_template---------')
    # print(' input , state = ', state, ', gripper = ', gripper)
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    #print('!!!!!!!1 tf.float32 , net -> ', net)
    net = tf.div(net, 255.)
    #print(' div 255 , net -> ', net)
    net = slim.conv2d(net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    #print(' conv2d 32, [8,8], stride=4 , net -> ', net)
    net = slim.conv2d(net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    #print(' conv2d 64, [4, 4], stride=2 , net -> ', net)
    net = slim.conv2d( net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    #print(' conv2d 64, [3, 3], stride=1 , net -> ', net)
    net = slim.flatten(net)

    
    net = slim.fully_connected( net, 512, weights_initializer=weights_initializer)

    print('before gripper net ', net)
    gripper = tf.cast(gripper, tf.float32)
    print('gripper ', gripper)
    net = tf.concat([net, gripper], axis=1)
    print('after gripper net ', net)

    #print(' flatten , net -> ', net)
    #print(' 512 , net -> ', net)
    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    print(' fully_connected , net -> ', net)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(q_values, logits, probabilities)


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
    print('in rainbow_rgb_gripper_ _build_networks')
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    # print(' self.online_convnet(self.state_ph)')
    self._net_outputs = self.online_convnet(self.state_ph, self.gripper_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    print('---before self.online_convnet(self._replay.states), self._replay.states = ', self._replay.states,', self._replay.gripper = ', self._replay.gripper)
    self._replay_net_outputs = self.online_convnet(self._replay.states, self._replay.gripper)
    print('---before self.target_convnet(self._replay.next_states)), self._replay.next_states = ', self._replay.next_states, ',self._replay.next_gripper=', self._replay.next_gripper)
    self._replay_next_target_net_outputs = self.target_convnet(self._replay.next_states, self._replay.next_gripper)


  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None,
                        gripper=0):
    # print('rainbow rgb gripper agent priority = ', priority)
    if priority is None:
      # print(' self._replay_scheme=',  self._replay_scheme ,', self._replay.memory.sum_tree.max_recorded_priority=', self._replay.memory.sum_tree.max_recorded_priority)
      priority = (1. if self._replay_scheme == 'uniform' else
                  self._replay.memory.sum_tree.max_recorded_priority)
    
    # print('rainbow rgb gripper agent priority after = ', priority)

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority, gripper)

  def step(self, reward, observation, gripper):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    # print('in rainbow_rgb_gripper step')
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False, gripper = gripper)
      self._train_step()

    self.action = self._select_action(gripper)
    return self.action


  def _select_action(self, gripper = 0):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
      int, the selected action.
    """
    # print('in rainbow_rgb_gripper _select_action')
    epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # print('self.state shape = ', np.shape(self.state), ', np.shape([gripper]) = ', np.shape([[gripper]]) )
      # Choose the action with highest Q-value at the current state.
      return self._sess.run(self._q_argmax, {self.state_ph: self.state, self.gripper_ph: [[gripper]]})

  def end_episode(self, reward, gripper):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward, True, gripper)