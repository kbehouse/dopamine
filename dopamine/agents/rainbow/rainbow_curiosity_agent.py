"""
Add Siamese Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow.rainbow_rgb_agent import RainbowRGBAgent
import numpy as np
import tensorflow as tf

import gin.tf
import random
import math
# slim = tf.contrib.slim
from dopamine.agents.rainbow.utils import conv, fc


def to2d(x):
  size = 1
  for shapel in x.get_shape()[1:]: size *= shapel.value
  return tf.reshape(x, (-1, size))

@gin.configurable
class RainbowCuriosityAgent(RainbowRGBAgent):
  """A compact implementation of a simplified Rainbow agent."""
  
  
  def _build_other(self):
    print('in RainbowCuriosityAgent _build_other()')
    self._build_curiosity(convfeat=32, rep_size=512, enlargement=2)
    # pass

  # from https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
  def _build_curiosity(self, convfeat, rep_size, enlargement, proportion_of_exp_used_for_predictor_update = 1.0):
    print("Using RND BONUS ****************************************************")
    print(f"convfeat={convfeat}, rep_size={rep_size}, enlargement={enlargement}")
    
    #RND bonus.

    # Random target network.
    # for ph in self.ph_ob.values():
    #   if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
    # print("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
    # xr = ph[:,1:]
    # print('after ph[:,1:] = xr ->', xr.shape)
    # xr = tf.cast(xr, tf.float32)
    # xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
    # print('after reshape = xr ->', xr.shape)
    # xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)
    # print('after clip_by_value xr ->', xr.shape)
    xr = tf.cast(self._replay.states, tf.float32)
    xr = tf.div(xr, 255.)
    print('xr -> ', xr)
    xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
    print('after c1r xr ->', xr.shape)
    xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
    print('after c2r xr ->', xr.shape)
    xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
    print('after c3r xr ->', xr.shape)

    rgbr = [to2d(xr)]
    print('rgbr ->', rgbr)

    X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))
    print('X_r-> ', X_r)
    print('after fc(rgbr[0] X_r ->', X_r.shape)

    # Predictor network.
    # for ph in self.ph_ob.values():
    #   if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
    # print("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
    # xrp = ph[:,1:]
    # xrp = tf.cast(xrp, tf.float32)
    # xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
    # xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)
    xrp = tf.cast(self._replay.states, tf.float32)
    xrp = tf.div(xrp, 255.)
    xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
    xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
    xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
    rgbrp = to2d(xrp)

    print('enlargement = ', enlargement)
    # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
    X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
    X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
    X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))
    print('X_r_hat.shape-> ', X_r_hat.shape)
    print('X_r_hat-> ', X_r_hat)


    self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
    self.max_feat = tf.reduce_max(tf.abs(X_r))
    self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
    # self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

    targets = tf.stop_gradient(X_r)
    # self.aux_loss = tf.reduce_mean(tf.square(noisy_targets-X_r_hat))
    tmp_squ = tf.square(targets - X_r_hat)
    print('tmp_squ -> ', tmp_squ)
    self.aux_loss = tf.reduce_mean(tf.square(targets - X_r_hat), -1)
    print('targets -> ', targets)
    print('self.aux_loss -> ', self.aux_loss.shape)
    mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
    mask = tf.cast(mask < proportion_of_exp_used_for_predictor_update, tf.float32)
    self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    print('self.aux_loss-> ', self.aux_loss)




  def _build_train_op(self):
    """Builds a training op.
    
    Returns:
      train_op: An op performing one step of training from replay data.
    """

    print('in RainbowCuriosityAgent _build_train_op()')
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))

      # Weight the loss by the inverse priorities.
      # loss = loss_weights * loss 
      ori_loss = loss_weights * loss 
      loss = ori_loss + self.aux_loss
      print('loss-> ', loss)
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('ori_loss', tf.reduce_mean(ori_loss))
          tf.summary.scalar('aux_loss', tf.reduce_mean(self.aux_loss))
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss