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
import modules.render as render
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances


import argparse
import pickle
import glob
import tensorflow as tf
import numpy as np
import pdb
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 3500,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 1000,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.001,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("input_directory", 'cifar-10-batches-py', "")
flags.DEFINE_string("summaries_dir", 'mnist_linear_logs','Summaries directory')
flags.DEFINE_boolean("relevance_bool", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_linear_model','Checkpoint dir')


FLAGS = flags.FLAGS

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def readData(input_directory, case='Train'):
    if case == 'Train':
        data = [unpickle(f) for f in glob.glob(input_directory+'/*') if 'data_batch' in f]
    else: 
        data =[unpickle(f) for f in glob.glob(input_directory+'/*') if 'test_batch' in f]
    images = np.vstack([d['data'] for d in data])
    img_shape = images.shape
    images = images.reshape(img_shape[0], 32,32,3)
    images = (images - 127.5) / 127.5
    labels = np.eye(10)[np.hstack([d['labels'] for d in data])]
    return images, labels

def batchGen(iterable1,iterable2, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

# def visualize(relevances, images_tensor):
#     n, w,h,c = relevances.shape
#     heatmap = relevances
#     input_images = images_tensor.reshape([n,28,28,1])
#     heatmaps = []
#     for h,heat in enumerate(heatmap):
#         input_image = input_images[h]
#         maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
#         heatmaps.append(maps)
#     R = np.array(heatmaps)
#     with tf.name_scope('input_reshape'):
#         img = tf.image_summary('input', tf.cast(R, tf.float32), n)
#     return img.eval()

# def init_vars(sess):
#     saver = tf.train.Saver()
#     tf.initialize_all_variables().run()
#     if FLAGS.reload_model:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             print('Reloading from -- '+FLAGS.checkpoint_dir+'/model.ckpt')
#             saver.restore(sess, ckpt.model_checkpoint_path)
#     return saver

# def save_model(sess, saver):
#     if FLAGS.save_model:
#         if not os.path.exists(FLAGS.checkpoint_dir):
#             os.system('mkdir '+FLAGS.checkpoint_dir)
#         save_path = saver.save(sess, FLAGS.checkpoint_dir+'/model.ckpt',write_meta_graph=False)

# def plot_relevances(rel, img, writer):
#     img_summary = visualize(rel, img)
#     writer.add_summary(img_summary)
#     writer.flush()


def nn():
    return Sequential([Convolution(input_dim=3,output_dim=32,input_shape=(FLAGS.batch_size, 32)), 
                       Relu(),
                       MaxPool(),
                       Convolution(input_dim=32,output_dim=64),
                       Relu(),
                       MaxPool(),
                       Linear(256, 256),
                       Relu(),
                       Linear(256, 10), 
                       Softmax()])


def train():
  # Import data
  with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32,32,3], name='x-input')
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    with tf.variable_scope('model'):
        net = nn()
        y = net.forward(x)
        train = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
        
    with tf.variable_scope('relevance'):   
        if FLAGS.relevance_bool:
            RELEVANCE = net.lrp(y, FLAGS.relevance_method, 1.0)
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
        
    train_imgs, train_labels = readData(FLAGS.input_directory, 'Train') 
    test_imgs, test_labels = readData(FLAGS.input_directory, 'Test') 
  
  
    for epoch in range(FLAGS.max_steps):
            
        for i, ii in enumerate(batchGen(train_imgs, train_labels, FLAGS.batch_size)):
            inp = {x: ii[0], y_:ii[1], keep_prob:0.8}
            summary, _ , relevance_train= sess.run([merged, train.train, RELEVANCE], feed_dict=inp)
            train_writer.add_summary(summary, i)
            
        test_acc = []
        #pdb.set_trace()
        for j, jj in enumerate(batchGen(test_imgs, test_labels, FLAGS.batch_size)):
            test_inp = {x: jj[0], y_:jj[1], keep_prob:1.0}
            summary, acc , relevance_test= sess.run([merged, accuracy, RELEVANCE], feed_dict=test_inp)
            test_acc.append(acc)
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %f' % (epoch, np.mean(test_acc)))

    # save model if required
    if FLAGS.save_model:
        utils.save_model()

    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance_bool:
        # plot test images with relevances overlaid
        images = jj[0]
        images = (images * 127.5) + 127.5
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,32,32,1]), images, test_writer )
        # plot train images with relevances overlaid
        images = ii[0]
        images = (images * 127.5) + 127.5
        plot_relevances(relevance_train.reshape([FLAGS.batch_size,32,32,1]), images, train_writer )

    train_writer.close()
    test_writer.close()







def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
