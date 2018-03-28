'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
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
from modules.avgpool import AvgPool
from modules.convolution import Convolution
import modules.render as render
from modules.utils import Utils, Summaries, plot_relevances
import input_data


import tensorflow as tf
import numpy as np
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 3501,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 1000,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_linear_logs','Summaries directory')
flags.DEFINE_boolean("relevance", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/epsilon/ww/flat/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS

def nn():
    return Sequential([Linear(input_dim=784,output_dim=1296, act ='relu', batch_size=FLAGS.batch_size, keep_prob=0.8),
                     Linear(1296, act ='relu'), 
                     Linear(1296, act ='relu'),
                     Linear(10, act ='relu'),
                       #Softmax()
    ])


# input dict creation as per tensorflow source code
def feed_dict(mnist, train):    
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        k = 1.0
    return (2*xs)-1, ys, k

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)

    # Model definition along with training and relevances
    with tf.variable_scope('model'):
        net = nn()
        y = net.forward(x)
        
        
    with tf.variable_scope('relevance'):    
        if FLAGS.relevance:
            LRP = net.lrp(y,FLAGS.relevance_method, 1)
            
            # LRP layerwise 
            relevance_layerwise = []
            R = y
            for layer in net.modules[::-1]:
                R = net.lrp_layerwise(layer, R, 'alphabeta',1)
                relevance_layerwise.append(R)
        else:
            LRP = []
            relevance_layerwise = []
            
    # Accuracy computation
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out 
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    
    utils = Utils(sess, FLAGS.checkpoint_dir)
    if FLAGS.reload_model:
        utils.reload_model()

    trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])

    uninit_vars = set(tf.global_variables()) - set(tf.trainable_variables())
    tf.variables_initializer(uninit_vars).run()
            
    # iterate over train and test data
    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:
            #pdb.set_trace()
            d = feed_dict(mnist, False)
            test_inp = {x:d[0], y_:d[1], keep_prob:d[2]}
            summary, acc , relevance_test, op, rel_layer= sess.run([merged, accuracy, LRP,y, relevance_layerwise], feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            print([np.sum(rel) for rel in rel_layer])
            print(np.sum(relevance_test))

            
        else:
            d = feed_dict(mnist, True)
            inp = {x:d[0], y_:d[1], keep_prob:d[2]}
            summary, _ , relevance_train,op, rel_layer= sess.run([merged, trainer.train, LRP,y, relevance_layerwise], feed_dict=inp)
            train_writer.add_summary(summary, i)
            
            
    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance:
        # plot test images with relevances overlaid
        images = d[0].reshape([FLAGS.batch_size,28,28,1])
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,28,28,1]), images, test_writer )
        # plot train images with relevances overlaid
        # images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        # images = (images + 1)/2.0
        # plot_relevances(relevance_train.reshape([FLAGS.batch_size,28,28,1]), images, train_writer )

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
