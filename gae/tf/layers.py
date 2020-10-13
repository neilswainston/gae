'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=wrong-import-order
import uuid

import numpy as np
import tensorflow as tf


class Layer():
    '''Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    '''

    def __init__(self, **kwargs):
        layer = self.__class__.__name__.lower()
        self.name = kwargs.get('name', layer + '_' + str(uuid.uuid4()))

        self.act = kwargs.get('act')
        self.dropout = kwargs.get('dropout')
        self.vars = {}
        self.logging = kwargs.get('logging', False)
        self.issparse = False

    def _call(self, inputs):
        '''Call.'''
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            return self._call(inputs)


class GraphConvolution(Layer):
    '''Graph convolution layer for undirected graph without edge labels.'''

    def __init__(self, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = _weight_variable_glorot(
                kwargs.get('input_dim'),
                kwargs.get('output_dim'),
                name='weights')

        self.adj = kwargs.get('adj')

    def _call(self, inputs):
        '''Call.'''
        x = tf.nn.dropout(inputs, rate=self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        return self.act(x)


def _weight_variable_glorot(input_dim, output_dim, name=''):
    '''Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.'''
    init_range = np.sqrt(6.0 / (input_dim + output_dim))

    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range)

    return tf.Variable(initial, name=name)
