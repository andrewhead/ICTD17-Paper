import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.preprocessing import image
from keras_models.vgg16 import VGG16
from keras.optimizers import SGD

import numpy as np

import math
import csv
from argparse import ArgumentParser
import os.path
from time import gmtime, strftime

from train import load_labels, load_test_indexes


# This method assumes that all labels are a string representing an integer
def load_labels(csv_filename):
    labels = []
    with open(csv_filename) as csvfile:
        rows = csv.reader(csvfile)
        first_row = True
        for row in rows:
            if first_row:
                first_row = False
                continue
            labels.append(row[6])
    return np.array(labels)


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
        examples_array = np.array(examples)
        repeats = math.ceil(float(sample_size) / len(examples))
        repeated_examples = np.array([])
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


# To the best of my ability, this creates the top layers of a neural network
# as demonstrated in the GitHub repository of Neal Jean at
# https://github.com/nealjean/predicting-poverty/blob/1b072cc418116332abfeea59fea095eaedc15d9a/model/predicting_poverty_deploy.prototxt
# This function returns the first layer and the last layer of that top.
def make_jean_top(num_classes=3):

    model = Sequential()
    model.add(Dropout(0.5, name="conv6_dropout", input_shape=(25, 25, 512)))
    model.add(Conv2D(
        filters=4096,
        kernel_size=(6, 6),
        strides=6,
        activation='relu',
        name="conv6",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    ))
    model.add(Dropout(0.5, name="conv7_dropout"))
    model.add(Conv2D(
        filters=4096,
        kernel_size=(1, 1),
        strides=1,
        activation='relu',
        name="conv7",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    ))
    model.add(Dropout(0.5, name="conv8_dropout"))
    model.add(Conv2D(
        filters=3,
        kernel_size=(1, 1),
        strides=1,
        name="conv8",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    ))
    model.add(AveragePooling2D(
        pool_size=(2, 2),
        strides=1,
        name="predictions_pooling"
    ))
    # XXX: I'm not sure this is correct (Neal's model may have created
    # a softmax for each pool individually) but it's good enough for now.
    model.add(Flatten(name="predictions_flatten"))
    model.add(Dense(num_classes, name="predictions_dense"))
    model.add(Activation('softmax', name="predictions"))
    
    return model


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


def train(features_dir, labels, test_indexes, batch_size, sample_size,
        learning_rate, epochs, kfolds, verbose=False, num_classes=3):

    if verbose:
        print("Building model of top of net...", end="")
    model = make_jean_top()
    if verbose:
        print("done.")

    if verbose:
        print("Compiling model...", end="")
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=SGD(lr=learning_rate),
        metrics=['accuracy'],
    )
    if verbose:
        print("done.")

    # Get list of indexes for all examples
    feature_files = os.listdir(features_dir)
    example_indexes = list(range(len(feature_files)))

    # Filter to the indexes that can be used for training
    training_indexes = list(filter(
        lambda i: i not in test_indexes, example_indexes))

    # Sample for equal representation of each class
    sampled_examples = sample_by_class(example_indexes, labels, sample_size)

    # Divide the sampled training data into folds
    folds = get_folds(sampled_examples, kfolds)

    # Convert labels to one-hot array for use in training.
    label_array = keras.utils.to_categorical(labels, num_classes)

    # Here, we fit the neural network for each fold
    for i, fold in enumerate(folds, start=1):

        training_examples = fold["training"]
        validation_examples = fold["validation"]

        if verbose:
            print("Training on fold %d of %d" % (i, len(folds)))
            print("Training set size: %d" % (len(training_examples)))
            print("Validation set size: %d" % (len(validation_examples)))

        # Do the actual fitting here
        model.fit_generator(
            FeatureExampleGenerator(training_examples, features_dir, label_array, batch_size),
            steps_per_epoch=math.ceil(float(len(training_examples)) / batch_size),
            epochs=epochs,
            verbose=(1 if verbose else 0),
            validation_data=FeatureExampleGenerator(validation_examples, features_dir, label_array, batch_size),
            validation_steps=math.ceil(float(len(validation_examples)) / batch_size),
        )
        if not os.path.exists("models"):
            os.makedirs("models")
        model.save(os.path.join(
            "models", "model-" + strftime("%Y%m%d-%H%M%S", gmtime()) + ".h5"))
 

if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument(
        "test_index_file",
        help="Name of file that has index of test sample on each line")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    argument_parser.add_argument(
        "--batch-size", default=16, type=int, help="Number of training examples at a time. " +
        "More than 16 at a time seems to lead to out-of-memory errors on K80")
    argument_parser.add_argument(
        "--sample-size", default=10000, type=int,
        help="Number of images to sample from each class (avoid biasing smaller classes).")
    argument_parser.add_argument("--learning-rate", default=0.01, type=float)
    argument_parser.add_argument("--epochs", default=10, type=int)
    argument_parser.add_argument("--num-folds", default=3, type=int)
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    labels = load_labels(args.csvfile)
    train(
        args.features_dir,
        labels,
        test_indexes,
        epochs=args.epochs,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        kfolds=args.num_folds,
        learning_rate=args.learning_rate,
        verbose=args.v,
    )
