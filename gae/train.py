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

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, mask_test_edges
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    '''main method.'''
    train()


def train(model_str='gcn_ae', dataset_str='cora', use_features=True,
          epochs=200, dropout=0.0, hidden1=32, hidden2=16, learning_rate=0.01):
    '''train.'''

    # Load data
    adj, features = load_data(dataset_str)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, _, val_edges, val_edges_false, test_edges, \
        test_edges_false = mask_test_edges(adj)
    adj = adj_train

    if not use_features:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model:
    model = None

    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero,
                           hidden1=hidden1, hidden2=hidden2)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features,
                            num_nodes, features_nonzero,
                            hidden1=hidden1, hidden2=hidden2)

    # Optimizer:
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / \
        float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(
                                  tf.sparse.to_dense(
                                      placeholders['adj_orig'],
                                      validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm,
                              learning_rate=learning_rate)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(
                                   tf.sparse.to_dense(
                                       placeholders['adj_orig'],
                                       validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               learning_rate=learning_rate)

    # Initialize session:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model:
    for epoch in range(epochs):
        t = time.time()

        # Construct feed dictionary:
        feed_dict = construct_feed_dict(
            adj_norm, adj_label, features, placeholders)

        feed_dict.update({placeholders['dropout']: dropout})

        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = _get_roc_score(
            feed_dict, placeholders, sess, model, adj_orig,
            val_edges, val_edges_false)

        val_roc_score.append(roc_curr)

        print('Epoch:', '%05d' % (epoch + 1),
              'train_loss=', '{:.5f}'.format(avg_cost),
              'train_acc=', '{:.5f}'.format(avg_accuracy),
              'val_roc=', '{:.5f}'.format(val_roc_score[-1]),
              'val_ap=', '{:.5f}'.format(ap_curr),
              'time=', '{:.5f}'.format(time.time() - t))

    roc_score, ap_score = _get_roc_score(
        feed_dict, placeholders, sess, model, adj_orig,
        test_edges, test_edges_false)

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


def _get_roc_score(feed_dict, placeholders, sess, model, adj_orig,
                   edges_pos, edges_neg, emb=None):
    '''Get ROC score.'''
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(_sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(_sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


if __name__ == '__main__':
    main()
