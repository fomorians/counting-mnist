from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def get_input_fn(filenames, batch_size, num_epochs):
    def _input_fn():
        with tf.device('/cpu:0'):
            filename_queue = tf.train.string_input_producer(filenames,
                                                            num_epochs=num_epochs)

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            features = tf.parse_single_example(serialized_example, features={
                'image': tf.FixedLenFeature([], tf.string),
                'density': tf.FixedLenFeature([], tf.string),
                'count': tf.FixedLenFeature([], tf.int64),
            })

            image = tf.decode_raw(features['image'], tf.uint8)
            density = tf.decode_raw(features['density'], tf.float32)
            count = features['count']

            image.set_shape([100 * 100])
            image = tf.reshape(image, [100, 100, 1])

            density.set_shape([100 * 100])
            density = tf.reshape(density, [100, 100, 1])

            image = tf.to_float(image) / 255
            image = image * 2 - 1

            images, densities, counts = tf.train.shuffle_batch(
                [image, density, count],
                batch_size=batch_size,
                num_threads=4,
                capacity=10000 + 3 * batch_size,
                min_after_dequeue=10000)

            counts = tf.expand_dims(counts, axis=-1)

            features = {
                'images': images
            }

            labels = {
                'densities': densities,
                'counts': counts,
            }

            return features, labels

    return _input_fn
