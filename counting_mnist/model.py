from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

NUM_CLASSES = 6 # 0-5

def count_parameters(scope=None):
    return np.sum([np.prod(tvar.get_shape().as_list()) \
                   for tvar in tf.contrib.framework.get_trainable_variables(scope=scope)])

def batch_norm(inputs, training):
    return tf.contrib.layers.batch_norm(
        inputs=inputs,
        is_training=training,
        decay=0.9,
        zero_debias_moving_mean=True,
        fused=True)

def get_logits(inputs, params, mode):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=8,
        kernel_size=7,
        strides=2,
        dilation_rate=1,
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=3,
        strides=1,
        dilation_rate=2,
        padding='same',
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=32,
        kernel_size=3,
        strides=1,
        dilation_rate=1,
        padding='same',
        activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=1,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)

    conv4_flat = tf.contrib.layers.flatten(conv4)

    dense4 = tf.layers.dense(
        inputs=conv4_flat,
        units=32,
        activation=tf.nn.relu)

    logits = tf.layers.dense(
        inputs=dense4,
        units=NUM_CLASSES,
        activation=None)
    return logits

def get_predictions(logits):
    predicted_counts = tf.argmax(logits, axis=-1)
    predictions = {
        'counts': predicted_counts,
    }
    return predictions

def get_loss(logits, counts, params, mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return None

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=counts,
        logits=logits)
    return loss

def get_train_op(loss, params, mode):
    train_op = None
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return train_op

    learning_rate = tf.constant(
        value=params['learning_rate'],
        dtype=tf.float32,
        name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    global_step = tf.contrib.framework.get_or_create_global_step()

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='Adam')

    return train_op

def model_fn(features, labels, params, mode):
    logits = get_logits(features['images'], params, mode)
    predictions = get_predictions(logits)

    loss = get_loss(logits, labels['counts'], params, mode)
    train_op = get_train_op(loss, params, mode)

    tf.summary.image('images', features['images'])
    tf.summary.image('densities', labels['densities'])

    tf.summary.histogram('counts', labels['counts'])
    tf.summary.histogram('predicted_counts', predictions['counts'])

    parameters = count_parameters()
    print('Parameters:', parameters)

    return predictions, loss, train_op
