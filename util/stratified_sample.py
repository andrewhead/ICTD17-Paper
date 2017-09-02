import numpy as np
import math
from argparse import ArgumentParser
import os.path

from util.load_data import load_labels


def sort_into_classes(example_indexes, labels):

    # Sort examples by class
    class_examples = {}
    for example_index in example_indexes:
        class_ = labels[example_index]
        if class_ not in class_examples:
            class_examples[class_] = []
        class_examples[class_].append(example_index)

    return class_examples


def get_example_indexes(features_dir):

    # Get list of indexes for all examples.  Assume that every file starts
    # with its index as its basename when making this list.
    example_indexes = []
    for feature_filename in os.listdir(features_dir):
        index = int(os.path.splitext(feature_filename)[0])
        example_indexes.append(index)

    return example_indexes


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Compute indexes for training set")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument("ratio", type=float, help="Proportion to put in validation set")
    argument_parser.add_argument(
        "max_samples_in_class",
        type=int,
        help="Max # of samples to include from any one category")
    argument_parser.add_argument(
        "output_training_index_file",
        help="Name of file to which to write training indexes")
    argument_parser.add_argument(
        "output_validation_index_file",
        help="Name of file to which to write validation indexes")
    args = argument_parser.parse_args()

    labels = load_labels(args.csvfile)
    examples = get_example_indexes(args.features_dir)
    class_examples = sort_into_classes(examples, labels)
    
    training_examples = []
    validation_examples = []

    for label, examples in class_examples.items():

        # Truncate to a max # of examples per class
        if len(examples) > args.max_samples_in_class:
            print("Too many examples in class", label, len(examples))
            examples = np.random.choice(examples, args.max_samples_in_class, replace=False)
            print("Now at this many", len(examples))

        # Shuffle examples so we get a random selection in each set
        np.random.shuffle(examples)

        # Split into trainin and validation for each class
        num_examples_in_class = len(examples)
        print("Num examples in class", len(examples))
        split_index = math.floor((1 - args.ratio) * num_examples_in_class)
        print("Making split at", split_index)
        training_examples.extend(examples[:split_index])
        validation_examples.extend(examples[split_index:])
        print("Now at size", len(training_examples), len(validation_examples))

    with open(args.output_training_index_file, 'w') as training_file:
        for example in training_examples:
            training_file.write(str(example) + "\n")

    with open(args.output_validation_index_file, 'w') as validation_file:
        for example in validation_examples:
            validation_file.write(str(example) + "\n")
