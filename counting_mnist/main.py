from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import argparse
import numpy as np
import tensorflow as tf

from counting_mnist.experiment import generate_experiment_fn

from tensorflow.contrib.learn.python.learn import learn_runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        default='data/counting_mnist/',
                        help='Counting MNIST data directory')
    parser.add_argument('--batch-size',
                        default=32,
                        type=int,
                        help='Batch size')
    parser.add_argument('--learning-rate',
                        default=1e-4,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--train-steps',
                        default=100000,
                        type=int,
                        help='Maximum number of training steps')
    parser.add_argument('--seed',
                        help='Random seed',
                        type=int,
                        default=random.randint(0, 2**32))
    parser.add_argument('--job-dir',
                        default='jobs/',
                        help='Job directory')
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    experiment_fn = generate_experiment_fn(args)
    learn_runner.run(experiment_fn, args.job_dir)

if __name__ == '__main__':
    main()
