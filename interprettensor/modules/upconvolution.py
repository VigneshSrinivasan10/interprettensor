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
from module import Module
import variables
import activations
import pdb

class Upconvolution(Module):
    '''
    Fractionally strided convolutional Layer /
    Convolutional transpose Layer
    '''

    def __init__(self, output_depth, input_depth=None, batch_size=None, input_dim=None, kernel_size=5, stride_size=2, act = 'linear', keep_prob=1.0, pad = 'SAME',weights_init= tf.truncated_normal_initializer(stddev=0.01), bias_init= tf.constant_initializer(0.0), name="deconv2d"):
        self.name = name
        Module.__init__(self)
        
        self.input_depth = input_depth
        self.input_dim = input_dim
        #self.check_input_shape()
        self.batch_size = batch_size
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.keep_prob = keep_prob
        self.act = act
        self.pad = pad

        self.weights_init = weights_init
        self.bias_init = bias_init
        
    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
            if len(inp_shape)!=4:
                mod_shape = [self.batch_size, self.input_dim,self.input_dim,self.input_depth]
                self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4; Specify input_dim in layer')
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.check_input_shape()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()
        self.input_depth = in_depth
        #pdb.set_trace()
        inp_shape = self.input_tensor.get_shape().as_list()
        if self.pad == 'SAME':
            output_shape = tf.stack([self.batch_size, inp_shape[1]*self.stride_size, inp_shape[1]*self.stride_size, self.output_depth])
        elif self.pad == 'VALID':
            output_shape = tf.stack([self.batch_size, (inp_shape[1]-1)*self.stride_size+self.kernel_size,(inp_shape[2]-1)*self.stride_size+self.kernel_size, self.output_depth])

        self.weights_shape = [self.kernel_size, self.kernel_size, self.output_depth, self.input_depth ]
        
        self.strides = [1,self.stride_size, self.stride_size,1]
        with tf.variable_scope(self.name):
            self.weights = variables.weights(self.weights_shape, initializer=self.weights_init, name=self.name)
            self.biases = variables.biases(self.output_depth, initializer=self.bias_init, name=self.name)
        
       
        with tf.name_scope(self.name):
            #pdb.set_trace()
            #deconv = tf.nn.atrous_conv2d(self.input_tensor, self.weights, rate=2, padding='SAME')
            deconv = tf.nn.conv2d_transpose(self.input_tensor, self.weights, output_shape=output_shape, strides = self.strides, padding=self.pad)
            deconv = tf.reshape(tf.nn.bias_add(deconv, self.biases), [-1]+deconv.get_shape().as_list()[1:])
            
            if isinstance(self.act, str): 
                self.activations = activations.apply(deconv, self.act)
            elif hasattr(self.act, '__call__'):
                self.activations = self.act(conv)

            if self.keep_prob<1.0:
                self.activations = tf.nn.dropout(self.activations, keep_prob=self.keep_prob)
            tf.summary.histogram('activations', self.activations)
            
        return self.activations

        
    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
            

        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (in_h -1) * hstride + hf - Hout
            pc =  (in_w -1) * wstride + wf - Wout
        
            # pr = in_h * hstride
            # pc =  in_w * wstride
            self.pad_input_tensor = tf.pad(self.R, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            pr = (in_h -1) * hstride + hf - Hout
            pc =  (in_w -1) * wstride + wf - Wout
            self.pad_input_tensor = tf.pad(self.R, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        
        #self.pad_input_tensor = self.input_tensor
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.input_tensor, dtype = tf.float32)
        #pdb.set_trace()
        out = []
        term1 = tf.expand_dims(self.weights, 0)
        
        term1 = tf.reshape(term1, [1,hf,wf,NF,df])
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(in_h):
            for j in xrange(in_w):
                input_slice = self.input_tensor[:, i:i+1,j:j+1, : ]
                term2 =  tf.expand_dims(input_slice, -1)
                
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                Zs = t1 + t2
                stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.pad_input_tensor[:,i*hstride:i*hstride+hf , j*wstride:j*wstride+wf,:], -2), 4)
                
                result = tf.reduce_sum(result, [1,2], keep_dims=True)
                # #pad each result to the dimension of the out
                pad_right = (in_h-1) - i 
                pad_left = i
                pad_bottom = (in_w - 1) -j 
                pad_up = j
                 
                re_N, re_h, re_W, re_depth = result.get_shape().as_list()
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                #pdb.set_trace()
                Rx+=result
        #pdb.set_trace()
        return Rx
        #return Rx[:, (pr/2):in_h+(pr/2), (pc/2):in_w+(pc/2),:]

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
            

        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        out = []
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.ones([N, hf,wf,NF], dtype=tf.float32)
                Zs = tf.reduce_sum(Z, [1,2,3], keep_dims=True) 
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 1), 4)
                
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
        return Rx[:, (pr/2):in_h+(pr/2), (pc/2):in_w+(pc/2),:]

    def _ww_lrp(self,R):
        '''
        LRP according to Eq(12) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
            

        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.square(tf.expand_dims(self.weights, 0))  
                Zs = tf.reduce_sum(Z, [1,2,3], keep_dims=True) 
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 1), 4)
                
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
        return Rx[:, (pr/2):in_h+(pr/2), (pc/2):in_w+(pc/2),:]

    def _epsilon_lrp(self,R, epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                Zs = t1 + t2
                stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 1), 4)
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
        return Rx[:, (pr/2):in_h+(pr/2), (pc/2):in_w+(pc/2),:]


    def _alphabeta_lrp(self,R, alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                Z = term1 * term2

                if not alpha == 0:
                    Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(tf.expand_dims(tf.where(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
                    t1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
                    Zsp = t1 + t2
                    Ralpha = alpha + tf.reduce_sum((Z / Zsp) * tf.expand_dims(self.R, 1),2)
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = tf.where(tf.lesser(Z,0),Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(tf.expand_dims(tf.where(tf.lesser(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
                    t1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
                    Zsp = t1 + t2
                    Rbeta = beta + tf.reduce_sum((Z / Zsp) * tf.expand_dims(self.R, 1),2)
                else:
                    Rbeta = 0

                result = Ralpha + Rbeta
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
        return Rx[:, (pr/2):in_h+(pr/2), (pc/2):in_w+(pc/2),:]
