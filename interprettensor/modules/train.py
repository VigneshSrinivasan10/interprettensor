'''
@author: Vignesh Srinivasan
@author: Sebastian Lapushkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf


class Train():
    def __init__(self, output=None,ground_truth=None,loss='softmax_crossentropy', optimizer='Adam', opt_params=[]):
        self.output = output
        self.ground_truth = ground_truth
        self.loss = loss
        self.optimizer = optimizer
        self.opt_params = opt_params

        self.learning_rate = self.opt_params[0]
        if len(self.opt_params)>1:
            self.var_list = self.opt_params[1]
        else:
            self.var_list = None

        if type(self.loss)!=str:
            #assuming loss is already computed and passed as a tensor
            self.cost = tf.reduce_mean(self.loss)
            #tf.summary.scalar('Loss', self.cost)
        else:
            self.compute_cost()
        
        self.optimize()
        
    def compute_cost(self):

        # Available loss operations are - [softmax_crossentropy, sigmoid_crossentropy, MSE] 
        if self.loss=='softmax_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.ground_truth)
                self.cost = tf.reduce_mean(diff)
            tf.summary.scalar('Loss', self.cost)

        elif self.loss=='sigmoid_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.ground_truth)
                self.cost = tf.reduce_mean(diff)
            tf.summary.scalar('Loss', self.cost)
                
        elif self.loss=='MSE':
            with tf.name_scope('mse_loss'):
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.output, self.ground_truth))))
            tf.summary.scalar('Loss', self.cost)
        else:
            print('Loss should be one of [softmax_crossentropy, sigmoid_crossentropy, MSE] ')
            print('If not define your own loss')
        
    

    def optimize(self):
        # Available loss operations are - [adam, adagrad, adadelta, grad_descent, rmsprop]
        with tf.name_scope('train'):
            if self.optimizer == 'adam':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            elif self.optimizer == 'rmsprop':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            elif self.optimizer == 'grad_descent':
                self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimizer == 'adagrad':
                self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.optimizer == 'adadelta':
                self.train = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)
            else:
                print('Optimizer should be one of: [adam, adagrad, adadelta, grad_descent, rmsprop]')

                
