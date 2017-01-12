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
from modules.upconvolution import Upconvolution as Tconvolution
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
flags.DEFINE_float("G_learning_rate", 0.001,'Initial learning rate')
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
    return Sequential([Convolution(input_depth=1,output_depth=32,batch_size=FLAGS.batch_size,input_dim=28), 
                     MaxPool(),
                     Tanh(),
                     Convolution(output_depth=64),
                     MaxPool(),
                     Tanh(),  
                     Linear(1)])
                     

def generator():
    #pdb.set_trace()
    return Sequential([Linear(input_dim=1024, output_dim=7*7*128, batch_size=FLAGS.batch_size),
                       Tanh(),
                       Convolution(input_dim=7,input_depth=128,output_depth=32), #4x4
                       Tanh(),
                       Tconvolution(output_depth=128, kernel_size=3), #8x8
                       Tanh(),
                       Tconvolution(output_depth=256, kernel_size=5, stride_size=1, pad='VALID'), #12x12
                       Tanh(),
                       Tconvolution(output_depth=32, kernel_size=3), #24X24 
                       Tanh(),
                       Tconvolution(output_depth=1, kernel_size=5, stride_size=1, pad='VALID'), #28X28
                       Tanh()])


def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.test_batch_size)
        k = 1.0
    return (2*xs)-1, ys, k


def compute_D_loss(D1, D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=D1, labels=tf.ones(tf.shape(D1))) , tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.zeros(tf.shape(D2)))

def compute_G_loss(D2):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones(tf.shape(D2)))

def visualize(relevances, images_tensor=None):
    n,w,h, dim = relevances.shape
    heatmaps = []
    if images_tensor is not None:
        assert relevances.shape==images_tensor.shape, 'Relevances shape != Images shape'
    for h,heat in enumerate(relevances):
        if images_tensor is not None:
            input_image = images_tensor[h]
            maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        else:
            maps = render.hm_to_rgb(heat, scaling = 3, sigma = 2)
        heatmaps.append(maps)
    R = np.array(heatmaps)
    return tf.cast(R, tf.float32)

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
            #pdb.set_trace()
            
            Gout = G.forward(tf.random_normal([FLAGS.batch_size, FLAGS.input_size]))
            packed = tf.concat(2, [Gout, tf.reshape(x, Gout.get_shape().as_list())])
            
            tf.summary.image('Generated-Original', packed, max_outputs=FLAGS.batch_size)
            #tf.summary.image('Original', tf.reshape(x, Gout.get_shape().as_list()))
            #pdb.set_trace()
            #with tf.variable_scope('model', reuse=True):
        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            D2 = D.forward(Gout)
            #pdb.set_trace()

    with tf.variable_scope('Loss'):
        total_params = tf.trainable_variables()
        D_params = total_params[:D_params_num]
        G_params = total_params[D_params_num:]
        D1_loss, D2_loss = compute_D_loss(D1, D2)
        D_loss = D1_loss + D2_loss
        G_loss = compute_G_loss(D2)
        tf.summary.scalar('D_real', tf.reduce_mean(D1_loss))
        tf.summary.scalar('D_fake', tf.reduce_mean(D2_loss))
        tf.summary.scalar('D_loss', tf.reduce_mean(D_loss))
        tf.summary.scalar('G_loss', tf.reduce_mean(G_loss))

        
    with tf.variable_scope('Trainer'):
        D_train = D.fit(loss=D_loss,optimizer='adam', opt_params=[FLAGS.D_learning_rate, D_params])
        G_train = G.fit(loss=G_loss,optimizer='adam', opt_params=[FLAGS.G_learning_rate, G_params])

    with tf.variable_scope('relevance'):
        if FLAGS.relevance_bool:
            D_RELEVANCE = D.lrp(D2, 'simple', 1.0)
            rel_packed = tf.concat(2, [Gout,D_RELEVANCE])
            
            #rel_imgs = visualize(D_RELEVANCE, Gout)
            tf.summary.image('Relevance-Disc', rel_packed, max_outputs=FLAGS.batch_size)
        else:
            D_RELEVANCE = []
        

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    D_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/D', sess.graph)
    G_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/G', sess.graph)

    
    
    tf.global_variables_initializer().run()
    utils = Utils(sess, FLAGS.checkpoint_dir)
    if FLAGS.reload_model:
        utils.reload_model()
    
    for i in range(FLAGS.max_steps):
        d = feed_dict(mnist, True)
        inp = {x:d[0]}
        #pdb.set_trace()
        D_summary, _ , dloss, dd1 ,dd2, relevance_train= sess.run([ merged, D_train.train, D_loss, D1_loss,D2_loss,D_RELEVANCE], feed_dict=inp)
        G_summary, _ , gloss, gen_images = sess.run([merged, G_train.train, G_loss, Gout], feed_dict=inp)
        G_summary, _ , gloss, gen_images = sess.run([merged, G_train.train, G_loss, Gout], feed_dict=inp)

        if i%100==0:
            print(gloss.mean(), dloss.mean())
            #pdb.set_trace()
        
        D_writer.add_summary(D_summary, i)
        G_writer.add_summary(G_summary, i)

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
