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
    '''Pre-process data.'''
    # Store original adjacency matrix (without diagonal entries) for later:
    adj_orig = adj

    adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)

    adj_orig.eliminate_zeros()

    adj, _, val_edges, val_edges_false, test_edges, \
        test_edges_false = _mask_test_edges(adj)

    # Some preprocessing:
    adj_norm = _preprocess_graph(adj)

    num_nodes = adj.shape[0]

    return adj, adj_orig, adj_norm, val_edges, val_edges_false, test_edges, \
        test_edges_false, num_nodes


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


def _preprocess_graph(adj):
    '''Preprocess graph.'''
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def _mask_test_edges(adj, prop_test=0.1, prop_val=0.05):
    '''Function to build test set with 10% positive links
    NOTE: Splits are randomized and results might slightly deviate from
    reported numbers in the paper.
    TODO: Clean up.'''

    # Remove diagonal elements:
    adj = adj - \
        sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * prop_test))
    num_val = int(np.floor(edges.shape[0] * prop_val))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack(
        [test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []

    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, \
        test_edges_false
