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
import scipy.sparse as sp


def preprocess_adj(adj):
    '''Pre-process adjacency.'''
    adj = sp.coo_matrix(adj)

    adj_ = adj + sp.eye(adj.shape[0])

    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())

    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_norm)


def preprocess_feat(features, use_features):
    '''Preprocess features.'''
    if not use_features:
        features = sp.identity(features.shape[0])  # featureless

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    num_nonzero_feats = features[1].shape[0]

    return features, num_features, num_nonzero_feats


def sparse_to_tuple(sparse_mx):
    '''Sparse to tuple.'''
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
