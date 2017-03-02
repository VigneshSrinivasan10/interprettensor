'''
@author: Vignesh Srinivasan
@author: Sebastian Lapushkin
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



from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class AvgPool(Module):

    def __init__(self, pool_size=(2,2), pool_stride=None, pad = 'SAME',name='avgpool'):
        self.name = name
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_size = [1]+list(self.pool_size)+[1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.pool_stride=self.pool_size
        self.pad = pad
        
    def forward(self,input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        with tf.name_scope(self.name):
            self.activations = tf.nn.avg_pool(self.input_tensor, ksize=self.pool_size,strides=self.pool_stride,padding=self.pad, name=self.name )
            tf.summary.histogram('activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    # def lrp(self,R,*args,**kwargs):
    #     return R

    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _, hf,wf,_ = self.pool_size
        _, hstride, wstride, _ = self.pool_stride
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        #import pdb; pdb.set_trace()
        Z = image_patches
        #Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
        #Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF])
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx

    def __simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            #similar to TF pad operation 
            pr = pr/2 
            pc = pc - (pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            

        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        #import pdb;pdb.set_trace()
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                #term1 = self.activations[:,i:i+1, j:j+1,:]
                Z = input_slice
                Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
                stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
                #pad each result to the dimension of the out
                pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx

    def _flat_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            #similar to TF pad operation 
            pr = pr/2 
            pc = pc - (pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            

        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        #import pdb;pdb.set_trace()
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.ones([N, hf,wf,NF], dtype=tf.float32)
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
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx

    def __flat_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            #similar to TF pad operation 
            pr = pr/2 
            pc = pc - (pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor    
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        #import pdb;pdb.set_trace()
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
        due to uniform weights used for sum pooling (1), this method defaults to _flat_lrp(R)
        '''
        return self._flat_lrp(R)

    def _epsilon_lrp(self,R,epsilon):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        _,hf,wf,_ = self.pool_size
        _,hstride, wstride,_= self.pool_stride

        out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            #similar to TF pad operation 
            pr = pr/2 
            pc = pc - (pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
        pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
                stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs)*-1, tf.ones_like(Zs)))
                Zs += stabilizer
                result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
                #pad each result to the dimension of the out
                pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
                   
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx           
        

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
        hf,wf,df,NF = self.pool_size
        _, hstride, wstride, _ = self.pool_stride

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                
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
                pad_right = pad_in_w - (i*hstride+hf) if( pad_in_w - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_h - (j*wstride+wf) if ( pad_in_h - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")                

                Rx+= result
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx           
        
