import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils.data_utils import get_file

import math
from argparse import ArgumentParser
from time import gmtime, strftime
import os.path

from util.load_data import load_labels, load_test_indexes
from util.sample import FeatureExampleGenerator


BLOCK5_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


# Sanity check to make sure that the first validation loss of a new epoch
# has actually changed relative to the end of the last epoch.
class StopIfValLossStationary(Callback):

    def __init__(self, val_loss):
        self.val_loss = val_loss

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if epoch == 0 and self.val_loss is not None and abs(current_val_loss - self.val_loss) < .00001:
            self.model.stop_training = True
            print("Val loss didn't change, stopping")


def train(features_dir, top_model_filename, labels, batch_size,
        learning_rate, momentum, epochs, training_indexes_filename,
        validation_indexes_filename, verbose=False, num_classes=3):

    START_TIMESTAMP = strftime("%Y%m%d-%H%M%S", gmtime())

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
    sgd = SGD(lr=learning_rate, momentum=momentum)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # Note: this learning rate should be pretty low (e.g., 1e-4, as
        # recommended in the referenced blog post, to keep previously-
        # learned features in tact.  Reference:
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        optimizer=sgd,
        metrics=['accuracy'],
    )
    if verbose:
        print("done.")

    # Load training and validation indexes from file
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

    if verbose:
        print("Training set size: %d" % (len(training_examples)))
        print("Validation set size: %d" % (len(validation_examples)))

    # Keep decreasing the learning rate until we reach a very small learning rate.
    # Each time, go until a maximum number of epochs or until the validation loss
    # stops noticeably decreasing.
    last_val_loss = None
    start_learning_rate = learning_rate
    while learning_rate >= .00001:

        # Do the actual fitting here
        history = model.fit_generator(
            FeatureExampleGenerator(training_examples, features_dir, label_array, batch_size),
            steps_per_epoch=math.ceil(float(len(training_examples)) / batch_size),
            # steps_per_epoch=1,
            epochs=epochs,
            # epochs=3,
            verbose=(1 if verbose else 0),
            validation_data=FeatureExampleGenerator(validation_examples, features_dir, label_array, batch_size),
            validation_steps=math.ceil(float(len(validation_examples)) / batch_size),
            # validation_steps=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=0),
                ModelCheckpoint(
                    get_best_model_filename(learning_rate, START_TIMESTAMP),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    period=1,                   
                ),
                StopIfValLossStationary(last_val_loss),
            ],
        )

        last_val_loss = history.history['val_loss'][-1]
        if verbose:
            print("Saving last val_loss", last_val_loss)
        if verbose:
            print("Loading best model from last round,", get_best_model_filename(learning_rate, START_TIMESTAMP), "...", end="")

        # Clear out existing models.  If we don't do this, we get an out-of-memory error
        # after loading a few models.
        K.clear_session()
        model = load_model(get_best_model_filename(learning_rate, START_TIMESTAMP))
        if verbose:
            print("done.")
        sgd = SGD(lr=learning_rate, momentum=momentum)
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=sgd,
            metrics=['accuracy'],
        )
        if verbose:
            print("Re-compiled model.")

        # Halve the learning rate for the next cycle
        learning_rate = learning_rate / 2
        if verbose:
            print("Had learning rate", K.get_value(sgd.lr), ", now changing to", learning_rate)
        K.set_value(sgd.lr, learning_rate)

    final_filename = os.path.join(
        "models",
        "tuned-finished.bs" + str(batch_size) + ".slr-" + str(start_learning_rate) + ".h5"
    )
    model.save(final_filename)
    return final_filename
 

def get_best_model_filename(learning_rate, timestamp):
    return os.path.join(
        'models', (
            'fine-tuned.' + timestamp +
            '.lr-' + str(learning_rate) + 
            '.h5'
        ))
 

if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("top_model", help="H5 for previously trained " +
        "top layers of the neural network.")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    argument_parser.add_argument(
        "--batch-size", default=16, type=int, help="Number of training examples at a time. " +
        "More than 16 at a time seems to lead to out-of-memory errors on K80")
    argument_parser.add_argument("--learning-rate", default=0.0001, type=float,
        help="(Should be low, to keep previously learned features in tact.)")
    argument_parser.add_argument("--momentum", default=0.9, type=float)
    argument_parser.add_argument("--epochs", default=50, type=int)
    argument_parser.add_argument("--training-indexes-file", help="File containing " +
        "an index of a training example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    argument_parser.add_argument("--validation-indexes-file", help="File containing " +
        "an index of a validation example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    labels = load_labels(args.csvfile)
    train(
        features_dir=args.features_dir,
        top_model_filename=args.top_model,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        training_indexes_filename=args.training_indexes_file,
        validation_indexes_filename=args.validation_indexes_file,
        verbose=args.v,
    )
