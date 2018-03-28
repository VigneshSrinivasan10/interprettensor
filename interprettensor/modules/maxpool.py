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


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class MaxPool(Module):

    def __init__(self, pool_size=2, pool_stride=None, pad = 'SAME',name='maxpool'):
        self.name = name
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_kernel = [1, self.pool_size, self.pool_size, 1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.stride_size=self.pool_size
        else:
            self.stride_size=self.pool_stride
        self.pool_stride=[1, self.stride_size, self.stride_size, 1] 
        self.pad = pad
        
    def forward(self,input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        self.in_N, self.in_h, self.in_w, self.in_depth = self.input_tensor.get_shape().as_list()
        
        with tf.name_scope(self.name):
            self.activations = tf.nn.max_pool(self.input_tensor, ksize=self.pool_kernel,strides=self.pool_stride,padding=self.pad, name=self.name )
            tf.summary.histogram('activations', self.activations)

        return self.activations

    def clean(self):
        self.activations = None
        self.R = None


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        self.check_shape(R)
        image_patches = self.extract_patches()
        Z = self.compute_z(image_patches)
        result = self.compute_result(Z)
        return self.restitch_image(result)

    def _epsilon_lrp(self,R, epsilon):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)

    def _ww_lrp(self,R): 
        '''
        There are no weights to use. default to _flat_lrp(R)
        '''
        return self._flat_lrp(R)
    
    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        self.check_shape(R)

        Z = tf.ones([self.in_N, self.Hout,self.Wout, self.pool_size,self.pool_size, self.in_depth])
        result = self.compute_result(Z)
        return self.restitch_image(result)
    
    def _alphabeta_lrp(self,R,alpha):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)
    
    def check_shape(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)
        N,self.Hout,self.Wout,NF = self.R.get_shape().as_list()

    def extract_patches(self):
        image_patches = tf.extract_image_patches(self.input_tensor, ksizes=[1, self.pool_size,self.pool_size, 1], strides=[1, self.stride_size,self.stride_size, 1], rates=[1, 1, 1, 1], padding=self.pad)
        return tf.reshape(image_patches, [self.in_N, self.Hout,self.Wout, self.pool_size,self.pool_size, self.in_depth])
        
    def compute_z(self, image_patches):
        Z = tf.equal( tf.reshape(self.activations, [self.in_N, self.Hout,self.Wout,1,1,self.in_depth]), image_patches)
        return tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
        
        
    def compute_zs(self, Z, stabilizer=True, epsilon=1e-12):
        Zs = tf.reduce_sum(Z, [3,4], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        if stabilizer==True:
            stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
            Zs += stabilizer
        return Zs

    def compute_result(self, Z,epsilon=1e-12):
        return Z * tf.reshape((self.R/(tf.reduce_sum(Z, [3,4])+epsilon)), [self.in_N,self.Hout,self.Wout,1,1,self.in_depth])
        
        
    def restitch_image(self, result):
        return self.patches_to_images(result, self.in_N, self.in_h, self.in_w, self.in_depth, self.Hout, self.Wout, self.pool_size, self.pool_size, self.stride_size,self.stride_size )




    def patches_to_images(self, grad, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c, stride_h, stride_r ):
        rate_r = 1
        rate_c = 1
        padding = self.pad
        
        
        ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
        ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

        if padding == 'SAME':
          if rows_out * 2 != rows_in:
            rows_out = int(ceil((rows_in+1) / stride_r))
            cols_out = int(ceil((cols_in+1) / stride_h))
          else:    
            rows_out = int(ceil(rows_in / stride_r))
            cols_out = int(ceil(cols_in / stride_h))
          pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
          pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

        elif padding == 'VALID':
          if rows_out * 2 != rows_in:
            rows_out = int(ceil(((rows_in+1) - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil(((cols_in+1) - ksize_c_eff + 1) / stride_h))
          else:    
            rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
          pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
          pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in


        
        pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

        grad_expanded = array_ops.transpose(
            array_ops.reshape(grad, (batch_size, rows_out,
                                     cols_out, ksize_r, ksize_c, channels)),
            (1, 2, 3, 4, 0, 5)
        )
        grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

        row_steps = range(0, rows_out * stride_r, stride_r)
        col_steps = range(0, cols_out * stride_h, stride_h)

        idx = []
        for i in range(rows_out):
            for j in range(cols_out):
                r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
                r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

                idx.extend([(r * (cols_in) + c,
                   i * (cols_out * ksize_r * ksize_c) +
                   j * (ksize_r * ksize_c) +
                   ri * (ksize_c) + ci)
                  for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                  for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                  if 0 <= r and r < rows_in and 0 <= c and c < cols_in
                ])

        sp_shape = (rows_in * cols_in,
              rows_out * cols_out * ksize_r * ksize_c)

        sp_mat = sparse_tensor.SparseTensor(
            array_ops.constant(idx, dtype=ops.dtypes.int64),
            array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
            sp_shape
        )

        jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

        grad_out = array_ops.reshape(
            jac, (rows_in, cols_in, batch_size, channels)
        )
        grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
        
        return grad_out
