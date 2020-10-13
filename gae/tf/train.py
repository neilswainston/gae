'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-order
import os
import time

from sklearn.metrics import average_precision_score, roc_auc_score

from gae.tf.model import get_model
from gae.tf.optimizer import get_opt
import numpy as np
import tensorflow as tf


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train(preprocess_adj, inner_product_decoder, get_adj_rec,
          adj, features, is_ae=True,
          epochs=64, dropout=0.0, num_hidden1=256, num_hidden2=128,
          learning_rate=0.01):
    '''train.'''

    # Adjacency:
    adj_norm = preprocess_adj(adj)

    # Get InnerProductDecoder:
    decoder = inner_product_decoder(
        act=lambda x: x,
        dropout=dropout,
        logging=True)

    # Create model:
    model = get_model(adj_norm, features, dropout, features.shape[-1],
                      num_hidden1, num_hidden2, decoder,
                      adj.shape[-2], is_ae)

    # Optimizer:
    opt = get_opt(model, adj.astype(np.float32), learning_rate, is_ae)

    # Initialize session:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train model:
    for epoch in range(epochs):
        t = time.time()

        with sess.as_default():
            print(model.hidden_layer1.shape,
                  model.hidden_layer1.eval().sum(),
                  model.vars['gcnmodelvae/hidden_layer1_vars/weights:0'].eval(
                  ).sum(),
                  adj.sum())

        # Run single weight update:
        _, avg_cost, avg_accuracy = sess.run(
            [opt.opt_op, opt.cost, opt.accuracy])

        # roc_curr, ap_curr = get_roc_score(
        #    adj_orig, val_edges, val_edges_false, adj_rec)

        print('Epoch:', '%05d' % (epoch + 1),
              'train_loss=', '{:.5f}'.format(avg_cost),
              'train_acc=', '{:.5f}'.format(avg_accuracy),
              # 'val_roc=', '{:.5f}'.format(roc_curr),
              # 'val_ap=', '{:.5f}'.format(ap_curr),
              'time=', '{:.5f}'.format(time.time() - t))

    adj_rec = get_adj_rec(sess, model)
    roc_score, ap_score = _get_roc_score(adj, adj_rec)

    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


def _get_roc_score(adj, adj_rec):
    '''Get ROC score.'''
    adj = adj.flatten()
    adj_rec = adj_rec.flatten()
    return roc_auc_score(adj, adj_rec), average_precision_score(adj, adj_rec)
