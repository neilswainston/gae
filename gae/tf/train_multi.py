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
import time

from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from gae.data import load_data
from gae.tf.layers import Layer
from gae.tf.model import get_model
from gae.tf.optimizer import get_opt
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


def train(adj, features, is_ae=True,
          epochs=200, dropout=0.0, num_hidden1=256, num_hidden2=128,
          learning_rate=0.01):
    '''train.'''

    # Adjacency:
    adj_norm = _preprocess_adj(adj)

    # Define placeholders:
    placeholders = {
        'adj': tf.compat.v1.placeholder(tf.float32),
        'features': tf.compat.v1.placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    # Get InnerProductDecoder:
    inner_product_decoder = InnerProductDecoder(
        act=lambda x: x,
        dropout=0.0,
        logging=True)

    # Create model:
    model = get_model(placeholders, features.shape[2],
                      num_hidden1, num_hidden2, inner_product_decoder,
                      adj.shape[1], is_ae)

    # Optimizer:
    adj_orig = (adj + np.eye(adj.shape[1])).astype(np.float32)
    opt = get_opt(model, adj, adj_orig, 1, learning_rate, is_ae)

    # Initialize session:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Construct feed dictionary:
    feed_dict = {
        placeholders['features']: features,
        placeholders['adj']: adj_norm,
        placeholders['dropout']: dropout
    }

    # Train model:
    for epoch in range(epochs):
        t = time.time()

        # Run single weight update:
        _, avg_cost, avg_accuracy = sess.run(
            [opt.opt_op, opt.cost, opt.accuracy],
            feed_dict=feed_dict)

        # roc_curr, ap_curr = get_roc_score(
        #    adj_orig, val_edges, val_edges_false, adj_rec)

        print('Epoch:', '%05d' % (epoch + 1),
              'train_loss=', '{:.5f}'.format(avg_cost),
              'train_acc=', '{:.5f}'.format(avg_accuracy),
              # 'val_roc=', '{:.5f}'.format(roc_curr),
              # 'val_ap=', '{:.5f}'.format(ap_curr),
              'time=', '{:.5f}'.format(time.time() - t))

    adj_rec = _get_adj_rec(sess, model, feed_dict)
    roc_score, ap_score = _get_roc_score(adj, adj_rec)

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


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

    return adj_norm


def _get_adj_rec(sess, model, feed_dict):
    '''Get reconstructed adjacency matrix.'''
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    emb_t = np.transpose(emb, axes=[0, 2, 1])
    adj_rec = expit(np.einsum('ijk,ikl->ijl', emb, emb_t))
    return (adj_rec + 0.5).astype(np.int)


def _get_roc_score(adj, adj_rec):
    '''Get ROC score.'''
    adj = adj.flatten()
    adj_rec = adj_rec.flatten()
    return roc_auc_score(adj, adj_rec), average_precision_score(adj, adj)


def main():
    '''main method.'''

    # Load data:
    adj, features = load_data('cora')

    # Train:
    train(np.array([adj.todense()]), np.array([features.todense()]),
          is_ae=False)


if __name__ == '__main__':
    main()
