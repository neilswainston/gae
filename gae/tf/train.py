'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
import os
import time

from gae.data import load_data
from gae.preprocessing import preprocess_adj, preprocess_feat, \
    sparse_to_tuple
from gae.results import get_roc_score
from gae.tf.model import get_model
from gae.tf.optimizer import get_opt
import scipy.sparse as sp
import tensorflow as tf


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train(adj, features, is_ae=True, use_features=True,
          epochs=200, dropout=0.0, hidden1=32, hidden2=16, learning_rate=0.01):
    '''train.'''

    # Adjacency:
    adj, adj_orig, adj_norm, val_edges, val_edges_false, test_edges, \
        test_edges_false, num_nodes = preprocess_adj(adj)

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
                      hidden1, hidden2, num_nodes, is_ae)

    # Optimizer:
    opt = get_opt(model, adj, placeholders['adj_orig'], num_nodes,
                  learning_rate, is_ae)

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

        emb = sess.run(model.z_mean, feed_dict=feed_dict)

        roc_curr, ap_curr = get_roc_score(
            adj_orig, val_edges, val_edges_false, emb)

        print('Epoch:', '%05d' % (epoch + 1),
              'train_loss=', '{:.5f}'.format(avg_cost),
              'train_acc=', '{:.5f}'.format(avg_accuracy),
              'val_roc=', '{:.5f}'.format(roc_curr),
              'val_ap=', '{:.5f}'.format(ap_curr),
              'time=', '{:.5f}'.format(time.time() - t))

    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    roc_score, ap_score = get_roc_score(
        adj_orig, test_edges, test_edges_false, emb)

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


def main():
    '''main method.'''

    # Load data:
    adj, features = load_data('cora')

    # Train:
    train(adj, features, is_ae=True)


if __name__ == '__main__':
    main()
