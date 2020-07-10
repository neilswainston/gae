'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
import tensorflow as tf


class OptimizerAE():
    '''AE optimiser.'''

    def __init__(self, preds, labels, pos_weight, norm, learning_rate):
        preds_sub = preds
        labels_sub = labels

        self.cost = self._get_cost(norm, preds, labels, pos_weight)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32))

        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def _get_cost(self, norm, preds, labels, pos_weight):
        '''Get cost.'''
        return norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds, labels=labels, pos_weight=pos_weight))


class OptimizerVAE():
    '''VAE optimiser.'''

    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,
                 learning_rate):
        self.model = model
        self.num_nodes = num_nodes
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, labels=labels_sub, pos_weight=pos_weight))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate)

        # Latent loss:
        kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                          tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
            tf.cast(labels_sub, tf.int32))

        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

    def _get_cost(self, norm, preds, labels, pos_weight):
        '''Get cost.'''
        # Latent loss:
        kl = (0.5 / self.num_nodes) * tf.reduce_mean(
            tf.reduce_sum(1 + 2 * self.model.z_log_std -
                          tf.square(self.model.z_mean) -
                          tf.square(tf.exp(self.model.z_log_std)), 1))

        return super._get_cost(norm, preds, labels, pos_weight) - kl
