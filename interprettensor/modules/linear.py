'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from modules.module import Module
import modules.variables as variables
import modules.activations as activations

class Linear(Module):
    '''
    Linear Layer
    '''

    def __init__(self, output_dim, batch_size=None, input_dim = None, act = 'linear', batch_norm = False, batch_norm_params = {'momentum':0.9, 'epsilon':1e-5, 'training':False,'name':'bn'}, keep_prob=tf.constant(1.0), weights_init= tf.truncated_normal_initializer(stddev=0.01), bias_init= tf.constant_initializer(0.0), name="linear"):
        self.name = name
        Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.act = act
        self.batch_norm = batch_norm
        self.batch_norm_params = batch_norm_params
        
        self.keep_prob = keep_prob

        self.weights_init = weights_init
        self.bias_init = bias_init
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        inp_shape = self.input_tensor.get_shape().as_list()

        if len(inp_shape)!=2:
            import numpy as np
            self.input_dim =  np.prod(inp_shape[1:])
            self.input_tensor = tf.reshape(self.input_tensor,[inp_shape[0], self.input_dim])
        else:
            self.input_dim = inp_shape[1]
        self.weights_shape = [self.input_dim, self.output_dim]
        with tf.name_scope(self.name):
            self.weights = variables.weights(self.weights_shape, initializer=self.weights_init, name=self.name)
            self.biases = variables.biases(self.output_dim, initializer=self.bias_init, name=self.name)

            
        with tf.name_scope(self.name):
            linear = tf.nn.bias_add(tf.matmul(self.input_tensor, self.weights), self.biases, name=self.name)
            if self.batch_norm:
                self.momentum = self.batch_norm_params['momentum']
                self.epsilon = self.batch_norm_params['epsilon']
                self.training = self.batch_norm_params['training']
                self.bn_name = self.batch_norm_params['name'] 
                linear = tf.contrib.layers.batch_norm(linear, decay=self.momentum, 
                                        updates_collections=None, epsilon=self.epsilon,
                                                      scale=True, is_training=self.training, scope=self.bn_name)
                                        
            if isinstance(self.act, str): 
                self.activations = activations.apply(linear, self.act)
            elif hasattr(self.act, '__call__'):
                self.activations = self.act(conv)

            def dropout_check_false():
                #print('Dropout adjusted 1.0')
                return tf.constant(1.0)
                
            def dropout_check_true():
                return tf.multiply(self.keep_prob, 1)
                
            # dropout_check = self.keep_prob<=tf.constant(1.0)
            
            # dropout = tf.cond(dropout_check, dropout_check_true, dropout_check_false)
            
            # self.activations = tf.nn.dropout(self.activations, keep_prob=dropout)
            #activations = activation_fn(conv, name='activation')
            tf.summary.histogram('activations', self.activations)
            tf.summary.histogram('weights', self.weights)
            tf.summary.histogram('biases', self.biases)
            
        return self.activations

    def check_input_shape(self):
        if len(self.input_shape)!=2:
            raise ValueError('Expected dimension of input tensor: 2')


    # def lrp(self, R):
    #     return self._simple_lrp(R)

    def clean(self):
        self.activations = None
        self.R = None

    def _simple_lrp(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=2:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)

        return self.input_tensor * tf.reduce_sum(tf.expand_dims(self.weights,0) * tf.expand_dims(self.R/(self.activations+1e-3),1), -1)
        
    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        self.R= R
        #import pdb;pdb.set_trace()
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=2:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)

        return self.input_tensor * tf.reduce_sum(tf.expand_dims(self.weights,0) * tf.expand_dims(self.R/(self.activations+epsilon),1), -1)

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        '''
        self.R= R
        Z = tf.ones_like(tf.expand_dims(self.weights, 0))
        Zs = tf.reduce_sum(Z, 1, keep_dims=True) 
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)
                         
    def _ww_lrp(self,R):
        '''
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        self.R= R
        Z = tf.square( tf.expand_dims(self.weights,0) )
        Zs = tf.expand_dims( tf.reduce_sum(Z, 1), 1)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1),2)
        
        
    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        self.R= R
        beta = 1 - alpha
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)

        if not alpha == 0:
            Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
            term1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
            Zsp = term1 + term2
            Ralpha = alpha * tf.reduce_sum((Zp / Zsp) * tf.expand_dims(self.R, 1),2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = tf.where(tf.less(Z,0),Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.less(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
            term1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
            Zsp = term1 + term2
            Rbeta = beta * tf.reduce_sum((Zn / Zsp) * tf.expand_dims(self.R, 1),2)
        else:
            Rbeta = 0

        return Ralpha + Rbeta
