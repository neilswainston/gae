'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=wrong-import-order
from gae.tf.sparse.layers_sparse import GraphConvolution, \
    GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf


def get_model(placeholders, dropout, num_features, num_nonzero_feats,
              hidden1, hidden2, num_nodes, is_ae):
    '''Get model.'''
    if is_ae:
        return GCNModelAE(placeholders['features'],
                          placeholders['adj'],
                          dropout,
                          num_features,
                          num_nonzero_feats,
                          hidden1=hidden1,
                          hidden2=hidden2)
    # else:
    return GCNModelVAE(placeholders['features'],
                       placeholders['adj'],
                       dropout,
                       num_features,
                       num_nodes,
                       num_nonzero_feats,
                       hidden1=hidden1, hidden2=hidden2)


class Model():
    '''Model baseclass.'''

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__.lower())
        self.logging = kwargs.get('logging', False)
        self.vars = {}

        self.z_mean = None
        self.reconstructions = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        ''' Wrapper for _build() '''
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        variables = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.vars = {var.name: var for var in variables}

    def fit(self):
        '''Fit.'''

    def predict(self):
        '''Predict.'''


class GCNModelAE(Model):
    '''GCN model autoencoder.'''

    def __init__(self, inputs, adj, dropout, num_features, num_nonzero_feats,
                 hidden1, hidden2, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = inputs
        self.input_dim = num_features
        self.num_nonzero_feats = num_nonzero_feats
        self.adj = adj
        self.dropout = dropout
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.build()

    def _build(self):
        hidden_layer1 = GraphConvolutionSparse(
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            adj=self.adj,
            num_nonzero_feats=self.num_nonzero_feats,
            act=tf.nn.relu,
            dropout=self.dropout,
            logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(hidden_layer1)

        self.reconstructions = InnerProductDecoder(
            act=lambda x: x,
            dropout=0.0,
            logging=self.logging)(self.z_mean)


class GCNModelVAE(Model):
    '''GCN model variational autoencoder.'''

    def __init__(self, inputs, adj, dropout, num_features, num_nodes,
                 num_nonzero_feats, hidden1, hidden2, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = inputs
        self.input_dim = num_features
        self.num_nonzero_feats = num_nonzero_feats
        self.n_samples = num_nodes
        self.adj = adj
        self.dropout = dropout
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.hidden_layer1 = None
        self.z = None
        self.z_mean = None
        self.z_log_std = None

        self.build()

    def _build(self):
        hidden_layer1 = GraphConvolutionSparse(
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            adj=self.adj,
            num_nonzero_feats=self.num_nonzero_feats,
            act=tf.nn.relu,
            dropout=self.dropout,
            logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(hidden_layer1)

        self.z_log_std = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(hidden_layer1)

        z = self.z_mean + \
            tf.random.normal(
                [self.n_samples, self.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(
            act=lambda x: x,
            dropout=0.0,
            logging=self.logging)(z)
