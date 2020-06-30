'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=wrong-import-order
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp


def parse_index_file(filename):
    '''Parse index file.'''
    index = []

    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    '''load the data: x, tx, allx, graph.'''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []

    for name in names:
        with open('data/ind.{}.{}'.format(dataset, name), 'rb') as fle:
            objects.append(pkl.load(fle, encoding='latin1'))

    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file(
        'data/ind.{}.test.index'.format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
