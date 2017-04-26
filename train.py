import keras
from keras import backend as K
from keras.models import Model
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
    label_vector = np.array(labels)
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

    WIDTH = 400
    HEIGHT = 400

    if verbose:
        print("Loading images...")

    X = np.zeros((len(image_paths), WIDTH, HEIGHT, 3))
    for img_index, img_path in enumerate(image_paths, start=0):
        img = image.load_img(img_path)
        img_array = image.img_to_array(img)
        X[img_index, :, :, :] = img_array
        if verbose and img_index > 0 and (img_index % 1000) == 0:
            print("Loaded %d of %d images..." % (img_index, len(image_paths)))

    if verbose:
        print("Loaded all images.")

    return X


# Written using the guidance from this Stack Overflow post:
# http://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy/41717938
def per_class_recall(class_id):

    def compute_recall(y_true, y_pred):
        true_classes = K.argmax(y_true, axis=-1)
        pred_classes = K.argmax(y_pred, axis=-1)
        recall_mask = K.cast(K.equal(true_classes, class_id), 'int32')
        classes_matching_target = K.cast(K.equal(true_classes, pred_classes), 'int32') * recall_mask
        recall = K.sum(classes_matching_target) / K.maximum(K.sum(recall_mask), 1)
        return recall
    
    # XXX: We use this hack of renaming the metric because Keras only shows the metrics
    # for a function with one name once (won't show this metric for 3 classes otherwise),
    # and this also makes the output look prettier.
    compute_recall.__name__ = "recall (C%d)" % class_id
    return compute_recall


def per_class_count_expected(class_id, batch_size):

    def compute_count_expected(y_true, y_pred):
        true_classes = K.argmax(y_true, axis=-1)
        expected_mask = K.cast(K.equal(true_classes, class_id), 'int32')
        return K.sum(expected_mask) / batch_size

    compute_count_expected.__name__ = "%% examples (C%d)" % class_id
    return compute_count_expected


def train(image_paths, y, num_classes=3, batch_size=32, epochs=10, kfolds=3, verbose=False):

    # Load baseline model (ImageNet)
    if verbose:
        print("Loading ImageNet model...", end="")
    model = VGG16(
        # Initialize with ImageNet weights
        weights="imagenet",
        # Continue training on 400x400 images.  We'll have to update the final
        # layers of the model to be fully convolutional.
        include_top=False,
        input_shape=(400, 400, 3),
    )
    if verbose:
        print("done.")

    if verbose:
        print("Updating final layers...", end="")

    # Add new fully convolutional "top" to the model
    # To the best of my ability, this follows the architecture published in
    # the GitHub repository of Neal Jean:
    # https://github.com/nealjean/predicting-poverty/blob/1b072cc418116332abfeea59fea095eaedc15d9a/model/predicting_poverty_deploy.prototxt
    # However, note that the VGG architecture that we initially load
    # varies from that described in Neal's `prototxt` file, even though we
    # try to make sure that the top layers are identical.
    layer = model.layers[-1].output
    layer = Dropout(0.5, name="conv6_dropout")(layer)
    layer = Conv2D(
        filters=4096,
        kernel_size=(6, 6),
        strides=6,
        activation='relu',
        name="conv6",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    )(layer)
    layer = Dropout(0.5, name="conv7_dropout")(layer)
    layer = Conv2D(
        filters=4096,
        kernel_size=(1, 1),
        strides=1,
        activation='relu',
        name="conv7",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    )(layer)
    layer = Dropout(0.5, name="conv8_dropout")(layer)
    layer = Conv2D(
        filters=3,
        kernel_size=(1, 1),
        strides=1,
        name="conv8",
        kernel_initializer=keras.initializers.glorot_normal(),
        bias_initializer=keras.initializers.Constant(value=0.1),
    )(layer)
    layer = AveragePooling2D(
        pool_size=(2, 2),
        strides=1,
        name="predictions_pooling"
    )(layer)
    # XXX: I'm not sure this is correct (Neal's model may have created
    # a softmax for each pool individually) but it's good enough for now.
    layer = Flatten(name="predictions_flatten")(layer)
    layer = Dense(num_classes, name="predictions_dense")(layer)
    layer = Activation('softmax', name="predictions")(layer)

    # Reset the model with the new top
    model = Model(model.input, layer)
    if verbose:
        print("done.")

    if verbose:
        print("Compiling model...", end="")
    # The `loss` came from the MNIST example (may be incorrect)
    # and the learning rate came from the Xie et al. paper,
    # "Transfer learning from deep features for remote sening and 
    # poverty mapping".
    metrics = ['accuracy']
    for class_index in range(num_classes):
        metrics.append(per_class_recall(class_index))
        metrics.append(per_class_count_expected(class_index, batch_size))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=SGD(lr=1e-6),
        metrics=metrics,
    )
    if verbose:
        print("done.")

    # Shuffle the images and labels
    index_order = np.array(list(range(len(image_paths))))
    np.random.shuffle(index_order)
    # X_shuffled = np.zeros(X.shape, dtype=K.floatx())
    image_paths_shuffled = np.zeros((len(image_paths),), dtype=object)
    y_shuffled = np.zeros(y.shape, dtype=K.floatx())
    for new_index, old_index in enumerate(index_order, start=0):
        image_paths_shuffled[new_index] = image_paths[old_index]
        y_shuffled[new_index] = y[old_index]

    image_paths = image_paths_shuffled
    y = y_shuffled

    # Train the model
    for fold_index in range(kfolds):

        # fold_size = math.ceil(len(X) / kfolds)
        fold_size = math.ceil(len(image_paths) / kfolds)
        val_fold_start = fold_size * fold_index
        val_fold_end = fold_size * (fold_index + 1)

        # Get the validation set
        # Using a trick from http://stackoverflow.com/questions/25330959/
        val_mask = np.zeros(len(image_paths), np.bool)
        val_mask[val_fold_start:val_fold_end] = 1
        image_paths_val = image_paths[val_mask]
        y_val = y[val_mask]

        # Get the training set
        train_mask = np.invert(val_mask)
        image_paths_train = image_paths[train_mask]
        y_train = y[train_mask]

        if verbose:
            print("Training on fold %d of %d" % (fold_index + 1, kfolds))
            print("Training set size: %d" % (len(image_paths_train)))
            print("Validation set size: %d" % (len(image_paths_val)))

        class Generator(object):

            def __init__(self, image_paths, labels, batch_size=32):
                self.index = 0
                self.image_paths = image_paths
                self.labels = labels
                self.batch_size = batch_size

            def __iter__(self):
                return self

            def __next__(self):
                return self.next()

            def next(self):

                # Retrieve next batch
                batch_image_paths = self.image_paths[self.index:self.index + self.batch_size]
                batch_labels = self.labels[self.index:self.index + self.batch_size]
                batch_images = load_images(batch_image_paths.tolist(), verbose=False)

                # Advance pointer for next batch
                self.index += batch_size
                if self.index >= len(self.image_paths):
                    self.index = 0

                return (batch_images, batch_labels)

        if verbose:
            print("Now fitting the model.")
        model.fit_generator(
            Generator(image_paths_train, y_train),
            steps_per_epoch=math.ceil(float(len(image_paths_train)) / batch_size),
            epochs=epochs,
            verbose=(1 if verbose else 0),
            validation_data=Generator(image_paths_val, y_val),
            validation_steps=math.ceil(float(len(image_paths_val)) / batch_size),
        )

        # Save the model after each fold
        model.save("model-" + strftime("%Y%m%d-%H%M%S", gmtime()) + ".h5")
 

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
    image_paths = get_image_paths(args.input_dir, test_indexes)
    y = load_labels(args.csvfile, test_indexes)

    train(image_paths, y, verbose=args.v)
