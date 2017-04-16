import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing import image
from keras_models.vgg16 import VGG16

import numpy as np

import math
import csv
from argparse import ArgumentParser
import os.path


def load_labels(csv_filename, test_indexes, num_classes=3):
    labels = []
    with open(csv_filename) as csvfile:
        rows = csv.reader(csvfile)
        first_row = True
        for row in rows:
            if first_row:
                first_row = False
                continue
            index = int(row[0])
            if index not in test_indexes:
                labels.append(row[6])
    label_vector = np.array(labels[:1000])
    label_array = keras.utils.to_categorical(label_vector, num_classes)
    return label_array


def load_test_indexes(test_index_filename):
    test_indexes = []
    with open(test_index_filename) as test_index_file:
        for line in test_index_file:
            test_indexes.append(int(line.strip()))
    return test_indexes


# It's assumed that the input directory contains nothing but the images for training.
def get_image_paths(input_dir, test_image_indexes):
    image_paths = []
    for filename in os.listdir(input_dir):
        image_index = int(filename.replace(".jpg", ""))
        if image_index not in test_image_indexes:
            image_paths.append(os.path.join(input_dir, filename))
    return image_paths


def load_images(image_paths, verbose=False):

    WIDTH = 224
    HEIGHT = 224

    if verbose:
        print("Loading images...")

    X = np.zeros((len(image_paths), WIDTH, HEIGHT, 3))
    for img_index, img_path in enumerate(image_paths, start=0):
        img = image.load_img(img_path, target_size=(WIDTH, HEIGHT))
        img_array = image.img_to_array(img)
        X[img_index, :, :, :] = img_array
        if verbose and img_index > 0 and (img_index % 1000) == 0:
            print("Loaded %d of %d images..." % (img_index, len(image_paths)))

    if verbose:
        print("Loaded all images.")

    return X


def train(X, y, num_classes=3, batch_size=16, epochs=12, kfolds=3, verbose=False):

    # Load baseline model (ImageNet)
    if verbose:
        print("Loading ImageNet model...", end="")
    model = VGG16()
    if verbose:
        print("done.")

    if verbose:
        print("Adding new final layer...", end="")

    # Remove previous classification layer
    model.layers.pop()

    # Create new classification layer
    last_feature_layer = model.layers[-1].output
    new_classification_layer = Dense(
        num_classes, activation='softmax', name='predictions')(last_feature_layer)

    # Name this to be a new model
    input_ = model.input
    model = Model(input_, new_classification_layer)
    if verbose:
        print("done.")

    # XXX: This is using the same compilation parameters as the
    # Keras MNIST example, which might not be appropriate.
    if verbose:
        print("Compiling model...", end="")
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    if verbose:
        print("done.")

    # Shuffle the images and labels
    index_order = np.array(list(range(len(X))))
    np.random.shuffle(index_order)
    X_shuffled = np.zeros(X.shape, dtype=K.floatx())
    y_shuffled = np.zeros(y.shape, dtype=K.floatx())
    for new_index, old_index in enumerate(index_order, start=0):
        X_shuffled[new_index] = X[old_index]
        y_shuffled[new_index] = y[old_index]

    # Train the model
    for fold_index in range(kfolds):

        fold_size = math.ceil(len(X) / kfolds)
        val_fold_start = fold_size * fold_index
        val_fold_end = fold_size * (fold_index + 1)

        # Get the validation set
        # Using a trick from http://stackoverflow.com/questions/25330959/
        val_mask = np.zeros(len(X), np.bool)
        val_mask[val_fold_start:val_fold_end] = 1
        X_val = X[val_mask]
        y_val = y[val_mask]

        # Get the training set
        train_mask = np.invert(val_mask)
        X_train = X[train_mask]
        y_train = y[train_mask]

        if verbose:
            print("Training on fold %d of %d" % (fold_index + 1, kfolds))
            print("Training set size: %d" % (len(X_train)))
            print("Validation set size: %d" % (len(X_val)))

        if verbose:
            print("Now fitting the model.")
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=(1 if verbose else 0),
            validation_data=(X_val, y_val),
        )


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Preprocess images for training")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument("input_dir")
    argument_parser.add_argument(
        "test_index_file",
        help="Name of file that has index of test sample on each line")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    image_paths = get_image_paths(args.input_dir, test_indexes)[:1000]
    X = load_images(image_paths, verbose=args.v)
    y = load_labels(args.csvfile, test_indexes)

    train(X, y, verbose=args.v)
