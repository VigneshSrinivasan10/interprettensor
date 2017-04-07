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
flags.DEFINE_string("checkpoint_dir", '/home/srinivasan/Projects/interprettensor/interprettensor/examples/mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS

def visualize(relevances, images_tensor):
    n, dim = relevances.shape
    heatmap = relevances.reshape([n,28,28,1])
    input_images = images_tensor.reshape([n,28,28,1])
    heatmaps = []
    for h,heat in enumerate(heatmap):
        input_image = input_images[h]
        maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.summary.image('input', tf.cast(R, tf.float32), n)
    return img.eval()


def init_vars(sess):
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    #tf.initialize_all_variables().run()
    #pdb.set_trace()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    try: 
        if ckpt and ckpt.model_checkpoint_path:
            print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tvars = np.load('/home/srinivasan/Projects/interprettensor/interprettensor/examples/mnist_linear_model/model.npy')
            for ii in range(8): sess.run(tf.trainable_variables()[ii].assign(tvars[ii]))
            pdb.set_trace()
            print('No model found!')
    except:
        raise ValueError('Layer definition and model layers mismatch!')
    return saver

def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
    writer.add_summary(img_summary)
    writer.flush()
    
def layers(x):
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
            my_netowrk = layers(x)
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
        plot_relevances(relevance_test, xs, test_writer)
        test_writer.close()
    
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    test()


if __name__ == '__main__':
    tf.app.run()
