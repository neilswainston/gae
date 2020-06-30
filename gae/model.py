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
from gae.layers import GraphConvolution, GraphConvolutionSparse, \
    InnerProductDecoder
import tensorflow as tf


class Model():
    '''Model baseclass.'''

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'Invalid keyword arg: ' + kwarg

        name = kwargs.get('name')

        if not name:
            name = self.__class__.__name__.lower()

        self.name = name

        self.logging = kwargs.get('logging', False)

        self.vars = {}

        self.hidden1 = None
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

    def __init__(self, placeholders, num_features, features_nonzero,
                 hidden1, hidden2, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.hidden_layer1 = None
        self.embeddings = None

        self.build()

    def _build(self):
        self.hidden_layer1 = GraphConvolutionSparse(
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout,
            logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(self.hidden_layer1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(
            act=lambda x: x,
            logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    '''GCN model variational autoencoder.'''

    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,
                 hidden1, hidden2, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.hidden_layer1 = None
        self.z = None
        self.z_mean = None
        self.z_log_std = None

        self.build()

    def _build(self):
        self.hidden_layer1 = GraphConvolutionSparse(
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            act=tf.nn.relu,
            dropout=self.dropout,
            logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(self.hidden_layer1)

        self.z_log_std = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            adj=self.adj,
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging)(self.hidden1)

        self.z = self.z_mean + \
            tf.random_normal(
                [self.n_samples, self.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(
            act=lambda x: x,
            logging=self.logging)(self.z)
