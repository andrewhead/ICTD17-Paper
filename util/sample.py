import numpy as np

import math
import csv
from argparse import ArgumentParser
import os.path
from time import gmtime, strftime

from util.load_data import load_labels, load_test_indexes


# Use this to get an equal representation of all classes in the
# set of examples that you'll be training on.
def sample_by_class(example_indexes, labels, sample_size):

    all_examples = np.array([], dtype=np.int32)

    # Sort examples by class
    class_examples = {}
    for example_index in example_indexes:
        class_ = labels[example_index]
        if class_ not in class_examples:
            class_examples[class_] = []
        class_examples[class_].append(example_index)

    # For each class...
    for class_, examples in class_examples.items():

        # Repeat the array as many times as it will take to get
        # enough examples for the sample.  Stack a bunch of shuffled
        # repeats on top of each other.  This sampling method lets us
        # avoid repeat sampling until all items have been sampled once.
        examples_array = np.array(examples, dtype=np.int32)
        repeats = math.ceil(float(sample_size) / len(examples))
        repeated_examples = np.array([], dtype=np.int32)
        for _ in range(repeats):
            repeat = examples_array.copy()
            np.random.shuffle(repeat)
            repeated_examples = np.concatenate((repeated_examples, repeat))

        # Truncate the repeated randomized lists to the sample size
        # and append to the shared list of output examples
        repeated_examples = repeated_examples[:sample_size]
        all_examples = np.concatenate((all_examples, repeated_examples))

    # Shuffle one more time at the end, as before this, all
    # examples have incidentally been sorted by class
    np.random.shuffle(all_examples)
    return all_examples


# Return a list of fold specs, where each one includes
# "train": a list of indexes of training examples
# "validation": a list of indexes of validation examples
# All test indexes will be omitted from all folds
def get_folds(example_indexes, num_folds=3):

    indexes_shuffled = np.array(example_indexes, dtype=np.int32)
    np.random.shuffle(indexes_shuffled)

    folds = []
    fold_size = math.ceil(len(indexes_shuffled) / num_folds)
    for fold_index in range(num_folds):

        fold_start = fold_size * fold_index
        fold_end = fold_start + fold_size

        validation_mask = np.zeros(len(indexes_shuffled), np.bool)
        validation_mask[fold_start:fold_end] = 1
        validation_indexes = indexes_shuffled[validation_mask]

        training_mask = np.invert(validation_mask)
        training_indexes = indexes_shuffled[training_mask]

        folds.append({
            "validation": validation_indexes,
            "training": training_indexes,
        })

    return folds


def get_training_examples(features_dir, labels, test_indexes, sample_size):

    # Get list of indexes for all examples.  Assume that every file starts
    # with its index as its basename when making this list.
    example_indexes = []
    for feature_filename in os.listdir(features_dir):
        index = int(os.path.splitext(feature_filename)[0])
        example_indexes.append(index)

    # Filter to the indexes that can be used for training
    training_indexes = list(filter(
        lambda i: i not in test_indexes, example_indexes))

    # Sample for equal representation of each class
    sampled_examples = sample_by_class(example_indexes, labels, sample_size)
    return sampled_examples


class FeatureExampleGenerator(object):

    def __init__(self, indexes, feature_dir, labels, batch_size):
        self.indexes = indexes
        self.feature_dir = feature_dir
        self.labels = labels
        self.batch_size = batch_size
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        # Get the list of example indexes in this batch
        batch_indexes = self.indexes[self.pointer:self.pointer + self.batch_size]

        # Load features for examples from file
        data_file_names = [
            os.path.join(self.feature_dir, str(i) + ".npz")
            for i in batch_indexes]
        examples = tuple()
        for filename in data_file_names:
            examples += (np.load(filename)["data"],)
        example_array = np.stack(examples)

        # Grab the labels for this batch
        labels = self.labels[batch_indexes]

        # Advance pointer for next batch
        self.pointer += self.batch_size
        if self.pointer >= len(self.indexes):
            self.pointer = 0

        return (example_array, labels)


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Compute indexes for training set")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument(
        "test_index_file",
        help="Name of file that has index of test sample on each line")
    argument_parser.add_argument(
        "sample_size", default=10000, type=int,
        help="Number of images to sample from each class (avoid biasing smaller classes).")
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    labels = load_labels(args.csvfile)
    examples = get_training_examples(
        features_dir=args.features_dir,
        labels=labels,
        test_indexes=test_indexes,
        sample_size=args.sample_size
    )
    for example in examples:
        print(example)

