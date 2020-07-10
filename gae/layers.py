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
from gae.initializations import weight_variable_glorot
import tensorflow as tf


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    '''Helper function, assigns unique layer IDs
    '''
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    # else:
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


class Layer():
    '''Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    '''

    def __init__(self, dropout, act, **kwargs):
        allowed_kwargs = {'name', 'logging'}

        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'Invalid keyword arg: ' + kwarg

        name = kwargs.get('name')

        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))

        self.act = act
        self.dropout = dropout
        self.name = name
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

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu,
                 **kwargs):
        super(GraphConvolution, self).__init__(dropout, act, **kwargs)

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')

        self.adj = adj

    def _call(self, inputs):
        '''Call.'''
        x = tf.nn.dropout(inputs, rate=self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        return self.act(x)


class GraphConvolutionSparse(GraphConvolution):
    '''Graph convolution layer for sparse inputs.'''

    def __init__(self, input_dim, output_dim, adj, num_nonzero_feats,
                 dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(input_dim, output_dim,
                                                     adj, dropout, act,
                                                     **kwargs)
        self.num_nonzero_feats = num_nonzero_feats

    def _call(self, inputs):
        '''Call.'''
        x = _dropout_sparse(inputs, self.dropout, self.num_nonzero_feats)
        x = tf.sparse.sparse_dense_matmul(x, self.vars['weights'])
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        return self.act(x)


class InnerProductDecoder(Layer):
    '''Decoder model layer for link prediction.'''

    def __init__(self, dropout=0.0, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(dropout, act, **kwargs)

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, rate=self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


def _dropout_sparse(x, rate, num_nonzero_elems):
    '''Dropout for sparse tensors. Currently fails for very large sparse
    tensors (>1M elements)'''
    keep_prob = 1 - rate
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob + tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)
