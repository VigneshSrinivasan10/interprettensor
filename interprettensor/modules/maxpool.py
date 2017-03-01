'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from module import Module


class MaxPool(Module):

    def __init__(self, pool_size=(2,2), pool_stride=None, pad = 'SAME',name='maxpool'):
        self.name = name
        Module.__init__(self)
        self.pool_size = [1]+ list(pool_size) + [1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.pool_stride=self.pool_size
        self.pad = pad
        
    def forward(self,input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        #with tf.variable_scope(self.name):
        with tf.name_scope(self.name):
            self.activations = tf.nn.max_pool(self.input_tensor, ksize=self.pool_size,strides=self.pool_stride,padding=self.pad, name=self.name )
            tf.summary.histogram('activations', self.activations)

        return self.activations

    def clean(self):
        self.activations = None

    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
            
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)
            

        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
        in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()
        
        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_rows
            pc =  (Wout -1) * wstride + wf - in_cols
            #similar to TF pad operation 
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor


        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                Z = tf.equal( self.activations[:,i:i+1, j:j+1,:], input_slice)
                Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
                Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
                result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
                #pad each result to the dimension of the out
                pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                
                Rx+= result
        
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_cols+(pc/2), (pr/2):in_rows+(pr/2), :]
        elif self.pad =='VALID':
            return Rx
        
    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)
            
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
        in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_rows
            pc =  (Wout -1) * wstride + wf - in_cols
            #similar to TF pad operation 
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor

        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.ones([N, hf,wf,NF], dtype=tf.float32)
                Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
                result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
                #pad each result to the dimension of the out
                pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_top = i*hstride
                pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_left = j*wstride
                result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
                
                Rx+= result
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx           

    def _ww_lrp(self,R):
        '''
        There are no weights to use. default to _flat_lrp(R)
        '''
        return self._flat_lrp(R)

    def _epsilon_lrp(self,R,epsilon):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)

    def _alphabeta_lrp(self,R,alpha):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)
