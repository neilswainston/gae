'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-name-in-module
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
import os

from scipy.special import expit

from gae.data import load_data
from gae.tf.layers import Layer
from gae.tf.train import train
import numpy as np
import tensorflow as tf


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class InnerProductDecoder(Layer):
    '''Decoder model layer for link prediction.'''

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, rate=self.dropout)
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


def do_train(adj, features, is_ae,
             epochs=64, dropout=0.0, num_hidden1=256, num_hidden2=128,
             learning_rate=0.01):
    '''Train.'''
    train(_preprocess_adj, InnerProductDecoder, _get_adj_rec,
          adj, features, is_ae,
          epochs=epochs, dropout=dropout,
          num_hidden1=num_hidden1, num_hidden2=num_hidden2,
          learning_rate=learning_rate)


def _preprocess_adj(adj):
    '''Pre-process adjacency.'''
    adj_ = adj + np.eye(adj.shape[1])

    rowsum = np.array(adj_.sum(2))

    degree_mat_inv_sqrt = np.array([np.diag(array)
                                    for array in np.power(rowsum, -0.5)])

    adj_norm = np.matmul(
        np.transpose(
            np.matmul(adj_, degree_mat_inv_sqrt),
            axes=(0, 2, 1)),
        degree_mat_inv_sqrt)

    return adj_norm.astype(np.float32)


def _get_adj_rec(sess, model):
    '''Get reconstructed adjacency matrix.'''
    emb = sess.run(model.z_mean)
    emb_t = np.transpose(emb, axes=[0, 2, 1])
    adj_rec = expit(np.einsum('ijk,ikl->ijl', emb, emb_t))
    return (adj_rec + 0.5).astype(np.int)


def main():
    '''main method.'''

    # Load data:
    adj, features = load_data('cora')

    # Train:
    do_train(np.array([adj.toarray()]), np.array([features.toarray()]),
             is_ae=False)


if __name__ == '__main__':
    main()
