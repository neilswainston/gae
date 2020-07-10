'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
import tensorflow as tf


def get_opt(model, adj, adj_orig, num_nodes, learning_rate, is_ae):
    '''Get optimiser.'''
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

    norm = adj.shape[0] * adj.shape[0] / \
        float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    with tf.name_scope('optimizer'):
        labels = tf.reshape(
            tf.sparse.to_dense(
                adj_orig,
                validate_indices=False), [-1])

        if is_ae:
            return OptimizerAE(preds=model.reconstructions,
                               labels=labels,
                               pos_weight=pos_weight,
                               norm=norm,
                               learning_rate=learning_rate)
        # else:
        return OptimizerVAE(preds=model.reconstructions,
                            labels=labels,
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm,
                            learning_rate=learning_rate)


class OptimizerAE():
    '''AE optimiser.'''

    def __init__(self, preds, labels, pos_weight, norm, learning_rate):
        self.cost = self._get_cost(norm, preds, labels, pos_weight)

        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate)

        self.opt_op = optimizer.minimize(self.cost)

        correct_prediction = tf.equal(
            tf.cast(tf.greater_equal(tf.sigmoid(preds), 0.5), tf.int32),
            tf.cast(labels, tf.int32))

        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

    def _get_cost(self, norm, preds, labels, pos_weight):
        '''Get cost.'''
        return norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds, labels=labels, pos_weight=pos_weight))


class OptimizerVAE(OptimizerAE):
    '''VAE optimiser.'''

    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,
                 learning_rate):
        self.model = model
        self.num_nodes = num_nodes
        super().__init__(preds, labels, pos_weight, norm, learning_rate)

    def _get_cost(self, norm, preds, labels, pos_weight):
        '''Get cost.'''
        # Latent loss:
        kl = (0.5 / self.num_nodes) * tf.reduce_mean(
            tf.reduce_sum(1 + 2 * self.model.z_log_std -
                          tf.square(self.model.z_mean) -
                          tf.square(tf.exp(self.model.z_log_std)), 1))

        return super()._get_cost(norm, preds, labels, pos_weight) - kl