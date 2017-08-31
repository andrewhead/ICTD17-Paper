import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.optimizers import SGD

import math
from argparse import ArgumentParser
import os.path

from util.load_data import load_labels
from util.sample import FeatureExampleGenerator


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


def train(features_dir, top_model_filename, labels, batch_size,
        learning_rate, epochs, training_indexes_filename,
        validation_indexes_filename, verbose=False, num_classes=3):

    if top_model_filename is not None:
        if verbose:
            print("Loading model of top of net...", end="")
        model = load_model(top_model_filename)
    else:
        if verbose:
            print("Building model of top of net...", end="")
        model = make_jean_top()
    if verbose:
        print("done.")

    if verbose:
        print("Compiling model...", end="")

    sgd = SGD(lr=learning_rate, momentum=0.9)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=sgd,
        metrics=['accuracy'],
    )
    if verbose:
        print("done.")

    # Sample for training indexes, or load from file
    training_examples = []
    validation_examples = []
    with open(training_indexes_filename) as training_indexes_file:
        for line in training_indexes_file:
            training_examples.append(int(line.strip()))
    with open(validation_indexes_filename) as validation_indexes_file:
        for line in validation_indexes_file:
            validation_examples.append(int(line.strip()))

    # Convert labels to one-hot array for use in training.
    label_array = keras.utils.to_categorical(labels, num_classes)

    # Only train for one of the fold, to better replicate Xie et al.
    if verbose:
        print("Training set size: %d" % (len(training_examples)))
        print("Validation set size: %d" % (len(validation_examples)))

    # Keep decreasing the learning rate until we reach a very small learning rate.
    # Each time, go until a maximum number of epochs or until the validation loss
    # stops noticeably decreasing.
    start_learning_rate = learning_rate
    while learning_rate >= .00001:

        # Do the actual fitting here
        model.fit_generator(
            FeatureExampleGenerator(training_examples, features_dir, label_array, batch_size),
            steps_per_epoch=math.ceil(float(len(training_examples)) / batch_size),
            epochs=epochs,
            verbose=(1 if verbose else 0),
            validation_data=FeatureExampleGenerator(validation_examples, features_dir, label_array, batch_size),
            validation_steps=math.ceil(float(len(validation_examples)) / batch_size),
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
        )

        save_model(model, batch_size, start_learning_rate, learning_rate)
        learning_rate = learning_rate / 2
        print("Had learning rate", K.get_value(sgd.lr), ", now changing to", learning_rate)
        K.set_value(sgd.lr, learning_rate)
 

def save_model(model, batch_size, start_learning_rate, learning_rate, suffix=""):
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save(os.path.join(
        "models", (
            "model-trained-top-" +
            "-bs-" + str(batch_size) +
            "-slr-" + str(start_learning_rate) +
            "-lr-" + str(learning_rate) +
            suffix + 
            ".h5"
        )
    ))


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    argument_parser.add_argument("--top-model", help="H5 for previously trained " +
        "top layers of the neural network.")
    argument_parser.add_argument(
        "--batch-size", default=16, type=int, help="Number of training examples at a time. " +
        "More than 16 at a time seems to lead to out-of-memory errors on K80")
    argument_parser.add_argument("--learning-rate", default=0.01, type=float)
    argument_parser.add_argument("--epochs", default=50, type=int)
    argument_parser.add_argument("--training-indexes-file", help="File containing " +
        "an index of a training example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    argument_parser.add_argument("--validation-indexes-file", help="File containing " +
        "an index of a validation example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    args = argument_parser.parse_args()

    labels = load_labels(args.csvfile)
    train(
        args.features_dir,
        args.top_model,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_indexes_filename=args.training_indexes_file,
        validation_indexes_filename=args.validation_indexes_file,
        verbose=args.v,
        batch_normalization=args.batch_normalization,
    )
