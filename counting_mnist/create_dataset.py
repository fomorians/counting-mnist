from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
import multiprocessing

import numpy as np
import tensorflow as tf

from functools import partial
from skimage.filters import gaussian

def get_distance(x, y, px, py):
    "Simple distance formula."
    return np.sqrt((px - x)**2 + (py - y)**2)

def get_points(num_points, xmin, ymin, xmax, ymax, min_distance):
    "Sample points from a 2D range with a minimum distance using rejection sampling."
    points = []

    for _ in range(num_points):
        while True:
            x = np.random.randint(xmin, xmax)
            y = np.random.randint(ymin, ymax)

            rejected = False
            for px, py in points:
                d = get_distance(x, y, px, py)
                if d < min_distance:
                    rejected = True
                    break

            if rejected:
                continue

            points.append((x, y))
            break

    return points

def generate_sample(dataset, image_size, digit_size, max_digits, sigma):
    "Construct a single counting MNIST sample."
    digit_size_half = digit_size // 2
    num_digits = np.random.randint(0, max_digits + 1)

    image = np.zeros(
        shape=(image_size, image_size, 1),
        dtype=np.float64)
    density = np.zeros(
        shape=(image_size, image_size, 1),
        dtype=np.float64)

    ids = np.random.choice(np.arange(dataset.num_examples), size=num_digits)
    digits = dataset.images[ids]
    labels = dataset.labels[ids]

    points = get_points(
        num_points=num_digits,
        xmin=digit_size_half,
        ymin=digit_size_half,
        xmax=image_size - digit_size_half,
        ymax=image_size - digit_size_half,
        min_distance=digit_size_half)

    count = 0
    for i, (digit, label, (x, y)) in enumerate(zip(digits, labels, points)):
        xmin = x - digit_size_half
        xmax = x + digit_size_half
        ymin = y - digit_size_half
        ymax = y + digit_size_half

        image[ymin:ymax, xmin:xmax] += digit

        if label % 2 == 0:
            density[y, x] = 1
            count += 1

    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)

    density = gaussian(density, sigma=sigma, mode='constant')
    density = density.astype(np.float32)

    return image, density, count

def generate_dataset(path, dataset, args):
    if os.path.exists(path):
        return

    writer = tf.python_io.TFRecordWriter(path)
    for i in range(args.num_set_examples):
        image, density, count = generate_sample(
            dataset=dataset,
            image_size=args.image_size,
            digit_size=args.digit_size,
            max_digits=args.max_digits,
            sigma=args.sigma)

        image_bytes = image.tobytes()
        density_bytes = density.tobytes()

        image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        density_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[density_bytes]))
        count_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))

        feature = {
            'image': image_feature,
            'density': density_feature,
            'count': count_feature,
        }

        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        example_str = example.SerializeToString()
        writer.write(example_str)

    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir',
                        default='data/mnist/',
                        help='MNIST source directory')
    parser.add_argument('--dest-dir',
                        default='data/counting_mnist/',
                        help='Counting MNIST destination directory')
    parser.add_argument('--num-set-examples',
                        default=100000,
                        type=int,
                        help='Number of examples per set')
    parser.add_argument('--num-train-sets',
                        default=10,
                        type=int,
                        help='Number of train sets containing 100k examples each')
    parser.add_argument('--num-test-sets',
                        default=1,
                        type=int,
                        help='Number of test sets containing 100k examples each')
    parser.add_argument('--image-size',
                        default=100,
                        type=int,
                        help='Size of one side of an image sample')
    parser.add_argument('--digit-size',
                        default=28,
                        type=int,
                        help='Size of a single digit in the image sample')
    parser.add_argument('--sigma',
                        default=5,
                        type=float,
                        help='Blur sigma for density map')
    parser.add_argument('--max-digits',
                        default=5,
                        type=int,
                        help='Maximum number of digits in an image sample')
    args, _ = parser.parse_known_args()

    print('Writing to {}...'.format(args.dest_dir))
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(args.source_dir, reshape=False)

    train_paths = [os.path.join(args.dest_dir, 'train_{}.tfrecords'.format(i + 1)) \
                   for i in range(args.num_train_sets)]

    test_paths = [os.path.join(args.dest_dir, 'test_{}.tfrecords'.format(i + 1)) \
                  for i in range(args.num_test_sets)]

    pool = multiprocessing.Pool()

    print('Writing train datasets...')
    generate_dataset_train = partial(generate_dataset, dataset=mnist.train, args=args)
    pool.map(generate_dataset_train, train_paths)

    print('Writing test datasets...')
    generate_dataset_test = partial(generate_dataset, dataset=mnist.test, args=args)
    pool.map(generate_dataset_test, test_paths)

if __name__ == '__main__':
    main()
