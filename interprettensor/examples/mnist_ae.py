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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.tconvolution import Tconvolution
import modules.render as render
from modules.utils import Utils, Summaries, plot_relevances
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 10000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("hidden_size", 10,'Number of steps to run trainer.')


flags.DEFINE_float("learning_rate", 0.001,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_ae_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_string("checkpoint_dir", 'mnist_ae_model','Checkpoint dir')


FLAGS = flags.FLAGS


def autoencoder(x):

    encoder = [Linear(784,500, input_shape=(FLAGS.batch_size,28)), 
                     Relu(),
                     Linear(500, 100), 
                     Relu(),
                     Linear(100, 10), 
                     ]
    decoder = [Linear(10,100), 
                     Relu(),
                     Linear(100, 500), 
                     Relu(),
                     Linear(500, 784),
                     Relu()
                     ]
    
    return Sequential(encoder+decoder)
    
    

def conv_autoencoder(x):

    encoder = [Convolution(input_dim=1,output_dim=32, input_shape=(FLAGS.batch_size,28)), 
                     Tanh(), 
                     Convolution(input_dim=32,output_dim=64),
                     Tanh(),  
                     Convolution(input_dim=64,output_dim=16),
                     Tanh(), 
                     Linear(256, 10)]
    decoder = [Tconvolution(input_dim=10,output_dim=128, kernel_size=(3,3), stride_size=(1,1)), 
                     Relu(),
               Tconvolution(input_dim=128,output_dim=64, kernel_size=(5,5)), 
                     Relu(),
               Tconvolution(input_dim=64,output_dim=32, kernel_size=(5,5), pad='VALID'), 
                     Relu(),
               Tconvolution(input_dim=32,output_dim=16, kernel_size=(5,5)), 
                     Relu(),
               Tconvolution(input_dim=16,output_dim=1, kernel_size=(5,5)), 
                     Relu()]
    
    return Sequential(encoder+decoder)


def noise(inp):
    noise = tf.get_variable(tf.truncated_normal(shape=[-1]+inp.get_shape().as_list()[1:], stddev = 0.1))
    return tf.addd(inp,noise)

# input dict creation as per tensorflow source code
def feed_dict(mnist, train):    
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
        k = 1.0
    return (2*xs)-1, ys, k

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    
    #noisy_x = noise(x)
    with tf.variable_scope('model'):
        #nn = autoencoder(x)
        net = conv_autoencoder(x)
        y = net.forward(x)
        output_shape = y.get_shape().as_list()
        #pdb.set_trace()
        y = tf.reshape(y, [FLAGS.batch_size, output_shape[1]*output_shape[2]*output_shape[3]])
        if FLAGS.relevance_bool:
            RELEVANCE = net.lrp(y, FLAGS.relevance_method, 1.0)
        else:
            RELEVANCE = []
        #pdb.set_trace()
        train = net.fit(output=y,ground_truth=x,loss='MSE',optimizer='adam', opt_params=[FLAGS.learning_rate])

        
    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    utils = Utils(sess, FLAGS.checkpoint_dir)
    if FLAGS.reload_model:
        utils.reload_model()
    

    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  # test-set accuracy
            d = feed_dict(mnist, False)
            test_inp = {x:d[0], y_:d[1], keep_prob:d[2]}
            summary, loss, relevance_test, test_op_imgs= sess.run([merged, train.cost, RELEVANCE,y], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Loss at step %s: %f' % (i, loss))
        else:
            d = feed_dict(mnist, True)
            inp = {x:d[0], y_:d[1], keep_prob:d[2]}
            summary, _ , relevance_train, op_imgs= sess.run([merged, train.train, RELEVANCE, y], feed_dict=inp)
            train_writer.add_summary(summary, i)
    #pdb.set_trace()
    if FLAGS.relevance_bool:
        # plot test images with relevances overlaid
        images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        images = (images + 1)/2.0
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,28,28,1]), images, test_writer )
        # plot train images with relevances overlaid
        images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        images = (images + 1)/2.0
        plot_relevances(relevance_train.reshape([FLAGS.batch_size,28,28,1]), images, train_writer )
        
        
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
