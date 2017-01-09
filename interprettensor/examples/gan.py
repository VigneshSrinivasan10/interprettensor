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
from modules.sigmoid import Sigmoid
from modules.convolution import Convolution
from modules.tconvolution import Tconvolution
import modules.render as render

from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import input_data

import argparse
import tensorflow as tf
import numpy as np
import pdb
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 5000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("input_size", 1024,'Number of steps to run trainer.')
flags.DEFINE_float("G_learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("D_learning_rate", 0.0001,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_gan_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_gan_model','Checkpoint dir')


FLAGS = flags.FLAGS


def discriminator():
    return Sequential([Convolution(input_dim=1,output_dim=32,input_shape=(FLAGS.batch_size, 28)), 
                     MaxPool(),
                     Tanh(),
                     Convolution(input_dim=32,output_dim=64),
                     MaxPool(),
                     Tanh(),  
                     Linear(256, 1)])
                     
                     #Softmax()
                   

def generator():
    return Sequential([Linear(1024, 7*7*128),
                       Tanh(),
                       Tconvolution(input_dim=7*7*128,output_dim=64, input_shape=(FLAGS.batch_size, 1)),
                       Tanh(),
                       Tconvolution(input_dim=64,output_dim=128, kernel_size=(3,3), stride_size=(1,1)),
                       Tanh(),
                       Tconvolution(input_dim=128,output_dim=256, kernel_size=(3,3), stride_size=(1,1)),
                       Tanh(),
                       Tconvolution(input_dim=256,output_dim=32, kernel_size=(3,3), stride_size=(1,1)),
                       Tanh(),
                       Tconvolution(input_dim=32,output_dim=16, pad='VALID'),
                       Tanh(),
                       Tconvolution(input_dim=16,output_dim=16),
                       Tanh(),
                       Tconvolution(input_dim=16,output_dim=1),
                       Tanh()
                       
                       ])


def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
        k = 1.0
    return (2*xs)-1, ys, k


def compute_D_loss(D1, D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(D1, tf.ones(tf.shape(D1))) , tf.nn.sigmoid_cross_entropy_with_logits(D2, tf.zeros(tf.shape(D2)))

def compute_G_loss(D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(D2, tf.ones(tf.shape(D2)))
    
def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 784], name='x-input')
        #pdb.set_trace()
    # Model definition along with training and relevances
    with tf.variable_scope('model'):
        with tf.variable_scope('discriminator'):
            D = discriminator()
            D1 = D.forward(x)
            D_params_num = len(tf.trainable_variables())
        with tf.variable_scope('generator'):
            G = generator()
            pdb.set_trace()
            
            Gout = G.forward(tf.random_normal([FLAGS.batch_size, FLAGS.input_size]))
            #pdb.set_trace()
    with tf.variable_scope('model', reuse=True):
        with tf.variable_scope('discriminator'):
            D2 = D.forward(Gout)

    total_params = tf.trainable_variables()
    D_params = total_params[:D_params_num]
    G_params = total_params[D_params_num:]

    D1_loss, D2_loss = compute_D_loss(D1, D2)
    D_loss = D1_loss + D2_loss
    D_train = D.fit(loss=D_loss,optimizer='adam', opt_params=[FLAGS.D_learning_rate, D_params])

    G_loss = compute_G_loss(D2)
    G_train = G.fit(loss=G_loss,optimizer='adam', opt_params=[FLAGS.G_learning_rate, G_params])
    
    if FLAGS.relevance_bool:
        D_RELEVANCE = D.lrp(D2, 'simple', 1.0)
    else:
        D_RELEVANCE = []
        

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    D_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/D')
    G_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/G')
    
    tf.global_variables_initializer().run()
    utils = Utils(sess, FLAGS.checkpoint_dir)
    if FLAGS.reload_model:
        utils.reload_model()
    
    for i in range(FLAGS.max_steps):
        d = feed_dict(mnist, True)
        inp = {x:d[0]}
        #pdb.set_trace()
        D_summary, _ , dloss, dd1 ,dd2, relevance_train= sess.run([ merged, D_train.train, D_loss, D1_loss,D2_loss,D_RELEVANCE], feed_dict=inp)
        _ , gloss, gen_images = sess.run([G_train.train, G_loss, Gout])
        _ , gloss, gen_images = sess.run([G_train.train, G_loss, Gout])

        if i%100==0:
            print(gloss.mean(), dloss.mean())
            #pdb.set_trace()
        
        D_writer.add_summary(D_summary, i)
        #G_writer.add_summary(G_summary, i)

    # save model if required
    if FLAGS.save_model:
        utils.save_model()
        
    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance_bool:
        # plot images with relevances overlaid
        images = (gen_images + 1)/2.0
        plot_relevances(relevance_train, images, D_writer )

    D_writer.close()
    G_writer.close()
    
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
