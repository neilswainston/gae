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
from gae.preprocessing_old import preprocess_adj, preprocess_feat, \
    sparse_to_tuple
from gae.tf.model import get_model
from gae.tf.optimizer import get_opt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train(adj, features, is_ae=True, use_features=True,
          epochs=200, dropout=0.0, hidden1=256, hidden2=128,
          learning_rate=0.01):
    '''train.'''

    # Adjacency:
    adj_norm = preprocess_adj(adj)

    # Features:
    features, num_features, num_nonzero_feats = \
        preprocess_feat(features, use_features)

    # Define placeholders:
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    # Create model:
    model = get_model(placeholders, num_features, num_nonzero_feats,
                      hidden1, hidden2, adj.shape[0], is_ae)

    # Optimizer:
    opt = get_opt(model, adj, placeholders['adj_orig'], learning_rate, is_ae)

    # Initialize session:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Construct feed dictionary:
    feed_dict = {
        placeholders['features']: features,
        placeholders['adj']: adj_norm,
        placeholders['adj_orig']: sparse_to_tuple(adj + sp.eye(adj.shape[0])),
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


def _get_adj_rec(sess, model, feed_dict):
    '''Get reconstructed adjacency matrix.'''
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    adj_rec = expit(np.dot(emb, emb.T))
    return (adj_rec + 0.5).astype(np.int)


def _get_roc_score(adj, adj_rec):
    '''Get ROC score.'''
    adj = adj.toarray().flatten()
    adj_rec = adj_rec.flatten()
    return roc_auc_score(adj, adj_rec), average_precision_score(adj, adj)


def main():
    '''main method.'''

    # Load data:
    adj, features = load_data('cora')

    # Train:
    train(adj, features, is_ae=False)


if __name__ == '__main__':
    main()
