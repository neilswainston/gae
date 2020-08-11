'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
import numpy as np


def preprocess_adj(adj):
    '''Pre-process adjacency.'''
    adj_ = adj + np.eye(adj.shape[0])

    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())

    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt)

    return adj_norm
