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



class Softmax(Module):
    '''
    Softmax Layer
    '''

    def __init__(self, name='softmax'):
        self.name = name
        Module.__init__(self)
    def forward(self,input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        with tf.name_scope(self.name):
            #with tf.name_scope('activations'):
            self.activations = tf.nn.softmax(self.input_tensor, name=self.name)
            tf.summary.histogram('activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        self.R = R
        Rx = self.R  * self.input_tensor
        #Rx = tf.nn.softmax(self.activations) * self.activations
        tf.summary.histogram(self.name, Rx)
        return Rx
    
    
