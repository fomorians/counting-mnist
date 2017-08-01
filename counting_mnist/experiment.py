from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import tensorflow as tf

from glob import glob

from counting_mnist.model import model_fn
from counting_mnist.data import get_input_fn

def generate_experiment_fn(args):
    def _experiment_fn(output_dir):
        train_filenames = glob(os.path.join(args.data_dir, 'train_*.tfrecords'))
        test_filenames = glob(os.path.join(args.data_dir, 'test_*.tfrecords'))

        train_input_fn = get_input_fn(train_filenames,
                                      batch_size=args.batch_size,
                                      num_epochs=None)
        eval_input_fn = get_input_fn(test_filenames,
                                     batch_size=args.batch_size,
                                     num_epochs=1)

        eval_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.accuracy,
                prediction_key='counts',
                label_key='counts'),
            'rmse': tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.root_mean_squared_error,
                prediction_key='counts',
                label_key='counts')
        }

        params = {
            'learning_rate': args.learning_rate,
        }

        config = tf.contrib.learn.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            save_checkpoints_secs=None)

        estimator = tf.contrib.learn.Estimator(
            model_fn=model_fn,
            model_dir=output_dir,
            params=params,
            config=config,
            feature_engineering_fn=None)

        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            eval_metrics=eval_metrics,
            train_steps=args.train_steps,
            eval_steps=None)

        return experiment
    return _experiment_fn
