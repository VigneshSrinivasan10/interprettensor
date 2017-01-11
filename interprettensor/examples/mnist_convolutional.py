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
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import input_data

import tensorflow as tf
import numpy as np
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 3001,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 500,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolution_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_convolution_model','Checkpoint dir')

FLAGS = flags.FLAGS


def nn():
    
    return Sequential([Convolution(output_depth=32,input_depth=1,batch_size=FLAGS.batch_size, input_dim=28),
                    Tanh(),
                    MaxPool(),
                    Convolution(64),
                    Tanh(),  
                    # Convolution(64),
                    # Tanh(),  
                    MaxPool(),
                    Linear(10), 
                    Softmax()])



def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        k = 1.0
    return (2*xs)-1, ys, k
    #return xs, ys, k


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('model'):
        net = nn()
        y = net.forward(x)
        train = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
    with tf.variable_scope('relevance'):
        if FLAGS.relevance_bool:
            #RELEVANCE = net.lrp(y, FLAGS.relevance_method, 1.0)
            RELEVANCE = net.lrp(y, 'simple', 1e-8)
            #RELEVANCE = net.lrp(y, 'ww', 0)
            #RELEVANCE = net.lrp(y, 'flat', 0)
            #RELEVANCE = net.lrp(y, 'alphabeta', 0.7)

            relevance_layerwise = []
            # R = y
            # for layer in net.modules[::-1]:
            #     R = net.lrp_layerwise(layer, R, 'simple')
            #     relevance_layerwise.append(R)

        else:
            RELEVANCE=[]
        
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
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
            test_inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            summary, acc , relevance_test, rel_layer= sess.run([merged, accuracy, RELEVANCE, relevance_layerwise], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            print([np.sum(rel) for rel in rel_layer])
            print(np.sum(relevance_test))
            #print(np.sum(op))
    
            # save model if required
            if FLAGS.save_model:
                utils.save_model()

        else:  
            d = feed_dict(mnist, True)
            inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            summary, _ , relevance_train,op, rel_layer= sess.run([merged, train.train, RELEVANCE,y, relevance_layerwise], feed_dict=inp)
            train_writer.add_summary(summary, i)
    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance_bool:
        # plot test images with relevances overlaid
        images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        #images = (images + 1)/2.0
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,28,28,1]), images, test_writer )
        # plot train images with relevances overlaid
        images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        #images = (images + 1)/2.0
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
