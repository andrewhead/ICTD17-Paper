import keras
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils.data_utils import get_file

import numpy as np

import math
from argparse import ArgumentParser
import os.path
from time import gmtime, strftime

from train import load_test_indexes
from train_top import load_labels, sample_by_class, get_folds, FeatureExampleGenerator


BLOCK5_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def train(features_dir, top_model_filename, labels, test_indexes, batch_size, sample_size,
        learning_rate, momentum, epochs, kfolds, verbose=False, num_classes=3):

    if verbose:
        print("Creating block 5 of VGG16..", end="")

    # Replication of block 5 of VGG16.  This is the layer that we're going
    # to retrain to become more attuned to daytime imagery.
    model = Sequential()
    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='block5_conv1',
        # Hard-coded input size.  This is the output size of `block4_pool` when
        # the input images are 400x400.
        input_shape=(25, 25, 512)
    ))
    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='block5_conv2'
    ))
    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        name='block5_conv3'
    ))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='block5_pool'
    ))

    if verbose:
        print("done.")

    # Initialize the layers of block 5 with the VGG16 ImageNet weights.
    # Note: we should load weights *before* adding the top of the model,
    # as we might clobber some of the previously trained weights in the
    # top if we load weights after the top has been added.
    if verbose:
        print("Loading ImageNet weights into block 5...", end="")
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        BLOCK5_WEIGHTS, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)
    if verbose:
        print("done.")

    # Load the previously trained top model, and add it to the top of the net.
    if verbose:
        print("Loading the top of the model from %s..." % (top_model_filename,), end="")
    top_model = load_model(top_model_filename)
    model.add(top_model)
    if verbose:
        print("done.")

    if verbose:
        print("Compiling model...", end="")
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # Note: this learning rate should be pretty low (e.g., 1e-4, as
        # recommended in the referenced blog post, to keep previously-
        # learned features in tact.  Reference:
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        optimizer=SGD(lr=learning_rate, momentum=momentum),
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
            "models", "tuned-" + strftime("%Y%m%d-%H%M%S", gmtime()) + ".h5"))
 

if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("top_model", help="H5 for previously trained " +
        "top layers of the neural network.")
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
    argument_parser.add_argument("--learning-rate", default=0.0001, type=float,
        help="(Should be low, to keep previously learned features in tact.)")
    argument_parser.add_argument("--momentum", default=0.9, type=float)
    argument_parser.add_argument("--epochs", default=10, type=int)
    argument_parser.add_argument("--num-folds", default=3, type=int)
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    labels = load_labels(args.csvfile)
    train(
        features_dir=args.features_dir,
        top_model_filename=args.top_model,
        labels=labels,
        test_indexes=test_indexes,
        epochs=args.epochs,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        kfolds=args.num_folds,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        verbose=args.v,
    )
