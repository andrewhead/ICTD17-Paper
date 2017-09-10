import keras
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator

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


def init_model(top_model_filename, learning_rate, momentum, verbose=False):

    if verbose:
        print("Re-creating VGG16 with regularization", end="")

    # Block 1
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(400, 400, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(5e-4)),
        MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(5e-4)),
        MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(5e-4)),
        MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(5e-4)),
        MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),
        # Block 5
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(5e-4)),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(5e-4)),
        MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),
    ])

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

    return model


def train(top_model_filename, old_model_filename, epochs, batch_size, learning_rate, momentum, verbose=False):

    START_TIMESTAMP = strftime("%Y%m%d-%H%M%S", gmtime())

    model = None
    if top_model_filename and not old_model_filename:
        model = init_model(top_model_filename, learning_rate, momentum, verbose=verbose)
    elif old_model_filename and not top_model_filename:
        model = load_model(old_model_filename)

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

    generator = ImageDataGenerator()
    training_data = generator.flow_from_directory(
        'images/Haiti_upsampled/training',
        target_size=(400, 400),
        batch_size=batch_size,
    )
    validation_data = generator.flow_from_directory(
        'images/Haiti_upsampled/validation',
        target_size=(400, 400),
        batch_size=batch_size,
    )

    # Keep decreasing the learning rate until we reach a very small learning rate.
    # Each time, go until a maximum number of epochs or until the validation loss
    # stops noticeably decreasing.
    last_val_loss = None
    start_learning_rate = learning_rate
    while learning_rate >= .0000000001:

        # Do the actual fitting here
        history = model.fit_generator(
            training_data,
            steps_per_epoch=30000 // batch_size,
            epochs=epochs,
            # epochs=3,
            validation_data=validation_data,
            validation_steps=3000 // batch_size,
            # steps_per_epoch=1,
            verbose=(1 if verbose else 0),
            # validation_steps=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=1),
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
        learning_rate = learning_rate / 10
        if verbose:
            print("Had learning rate", K.get_value(sgd.lr), ", now changing to", learning_rate)
        K.set_value(sgd.lr, learning_rate)

    final_filename = os.path.join(
        "models",
        "long-tuned.bs" + str(batch_size) + ".slr-" + str(start_learning_rate) + ".h5"
    )
    model.save(final_filename)
    return final_filename


def get_best_model_filename(learning_rate, timestamp):
    return os.path.join(
        'models', (
            'long-tuned.' + timestamp +
            '.lr-' + str(learning_rate) + 
            '.h5'
        ))
 

if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train the whole neural net")
    argument_parser.add_argument("--top-model", help="H5 for previously trained " +
        "top layers of the neural network.")
    argument_parser.add_argument("--old-model", help="H5 for previously trained " +
        "entire network.")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    argument_parser.add_argument(
        "--batch-size", default=32, type=int, help="Number of training examples at a time. " +
        "More than 16 at a time seems to lead to out-of-memory errors on K80")
    argument_parser.add_argument("--learning-rate", default=0.000001, type=float,
        help="(Should be low, to keep previously learned features in tact.)")
    argument_parser.add_argument("--momentum", default=0.9, type=float)
    argument_parser.add_argument("--epochs", default=100, type=int)
    args = argument_parser.parse_args()

    train(
        top_model_filename=args.top_model,
        old_model_filename=args.old_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        verbose=args.v,
    )
