import keras
from keras.models import Sequential, save_model, load_model
from keras.optimizers import SGD
from keras.layers import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from argparse import ArgumentParser


def flatten(tuned_model_filename, output_filename):

    # Load the top of the past model.  We'll need this for manually
    # transferring over the weights of each of those layers.
    tuned_model = load_model(tuned_model_filename)
    tuned_model_top = tuned_model.layers[-1]

    # XXX: Here we clone the model architecture.  `tune_block5.py` creates
    # a hierarchical version of this, which we want to flatten so that we
    # can easily access the output of intermediate layers.
    # Replication of VGG16 (See `tune_whole.py`)
    # Block 1
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(400, 400, 3), activation='relu', padding='same', name='block1_conv1'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),
        # Block 5
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),
    ])
    # This is a replication of the prediction layers from Jean et al.
    # Originally defined in `train_top.py` file.
    conv6_layer_index = len(model.layers)
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
    model.add(Flatten(name="predictions_flatten"))
    model.add(Dense(3, name="predictions_dense"))
    model.add(Activation('softmax', name="predictions"))

    # This loads weights into the block5 layers.
    for layer_index in range(conv6_layer_index):
        model.layers[layer_index].set_weights(
            tuned_model.layers[layer_index].get_weights())

    # We still need to manually load weights into all the layers after block5
    # which were nested in the original tuned model.
    for layer_index in range(len(tuned_model_top.layers)):
        model.layers[conv6_layer_index + layer_index].set_weights(
            tuned_model_top.layers[layer_index].get_weights())

    # We compile this here so that when we load it later, it will already
    # be compiled (which our `extract_features` script expects).  Though
    # this step could be taken out if we don't need to reload pre-compiled
    # models in later scripts.
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=SGD(lr=0.0001, momentum=0.9),
        metrics=['accuracy'],
    )

    model.save(output_filename)


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Flatten tuned net." +
        "The net is expected to start a block 4 of VGG16 and end with " +
        "predictions from the top we train for nightlights prediction.")
    argument_parser.add_argument("tuned_model",
        help="Name of file containing hierarchical tuned model")
    argument_parser.add_argument("output_file",
        help="Name of file to write flattened model to")
    args = argument_parser.parse_args()
    flatten(args.tuned_model, args.output_file)
