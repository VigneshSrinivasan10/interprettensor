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
import modules.render as render
import input_data
from modules.utils import Utils, Summaries, plot_relevances

import argparse
import tensorflow as tf
import numpy as np
import pdb
import os

flags = tf.flags
logging = tf.logging



flags.DEFINE_integer("batch_size", 200,'Number of steps to run trainer.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'my_model_logs','Summaries directory')
flags.DEFINE_boolean("relevance", True,'Compute relevances')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS


def init_vars(sess):
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    try: 
        if ckpt and ckpt.model_checkpoint_path:
            print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tvars = np.load('mnist_linear_model/model.npy')
            for ii in range(8): sess.run(tf.trainable_variables()[ii].assign(tvars[ii]))
            print('Reloading from numpy array!')
    except:
        raise ValueError('Layer definition and model layers mismatch!')
    return saver
    
def layers():
    # Define the layers of your network here
    
    return Sequential([Linear(input_dim=784,output_dim=1296, act='relu', batch_size=FLAGS.batch_size),                    
                     Linear(1296, act='relu'), 
                     Linear(1296, act='relu'),
                     Linear(10),
                     Softmax()])


def test():

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 784], name='input')
        with tf.variable_scope('model'):
            my_netowrk = layers()
            output = my_netowrk.forward(x)
            if FLAGS.relevance:
                RELEVANCE = my_netowrk.lrp(output, 'simple', 1.0)
                
        # Merge all the summaries and write them out 
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/my_model')

        # Intialize variables and reload your model
        saver = init_vars(sess)
        
        # Extract testing data 
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        # Pass the test data to the restored model
        summary, relevance_test= sess.run([merged, RELEVANCE], feed_dict={x:(2*xs)-1})
        test_writer.add_summary(summary, 0)

        # Save the images as heatmaps to visualize on tensorboard
        images = xs.reshape([FLAGS.batch_size,28,28,1])
        images = (images + 1)/2.0
        relevances = relevance_test.reshape([FLAGS.batch_size,28,28,1])
        plot_relevances(relevances, images, test_writer )

        test_writer.close()
    
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
