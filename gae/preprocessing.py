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
    # adj = sp.coo_matrix(adj)

    adj_ = adj + np.eye(adj.shape[1])

    rowsum = np.array(adj_.sum(2))

    degree_mat_inv_sqrt = np.array([np.diag(array)
                                    for array in np.power(rowsum, -0.5)])

    adj_norm = np.matmul(
        np.transpose(
            np.matmul(adj_, degree_mat_inv_sqrt),
            axes=(0, 2, 1)),
        degree_mat_inv_sqrt)

    return sparse_to_tuple(adj_norm)


def preprocess_feat(features):
    '''Preprocess features.'''
    features = sparse_to_tuple(features)
    num_features = features[2][2]
    num_nonzero_feats = features[1].size

    return features, num_features, num_nonzero_feats


def sparse_to_tuple(matrix):
    '''Sparse to tuple.'''
    nonzero = matrix.nonzero()
    nonzero_t = np.transpose(nonzero)

    coords = np.array(
        np.split(nonzero_t[:, (1, 2)],
                 np.cumsum(np.unique(nonzero_t[:, 0],
                                     return_counts=True)[1])[:-1]))

    values = matrix[nonzero].reshape(matrix.shape[0], -1)
    return coords, values, matrix.shape
