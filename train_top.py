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

from util.load_data import load_labels, load_test_indexes
from util.sample import get_folds, get_training_examples, FeatureExampleGenerator


# To the best of my ability, this creates the top layers of a neural network
# as demonstrated in the GitHub repository of Neal Jean at
# https://github.com/nealjean/predicting-poverty/blob/1b072cc418116332abfeea59fea095eaedc15d9a/model/predicting_poverty_deploy.prototxt
# This function returns the first layer and the last layer of that top.
def make_jean_top(num_classes=3):

    model = Sequential()
    model.add(Dropout(0.5, name="conv6_dropout", input_shape=(12, 12, 512)))
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


def train(features_dir, labels, test_indexes, batch_size, sample_size,
        learning_rate, epochs, kfolds, training_indexes_filename,
        verbose=False, num_classes=3):

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

    # Sample for training indexes, or load from file
    if training_indexes_filename is not None:
        sampled_examples = []
        with open(training_indexes_filename) as training_indexes_file:
            for line in training_indexes_file:
                sampled_examples.append(int(line.strip()))
    else:
        sampled_examples = get_training_examples(
            features_dir, labels, test_indexes, sample_size)

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
    argument_parser.add_argument("--training-indexes-file", help="File containing " +
        "an index of a training example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
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
        training_indexes_filename=args.training_indexes_file,
        verbose=args.v,
    )
