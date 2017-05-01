from keras.models import load_model
from keras.utils import to_categorical

import numpy as np
from sklearn.metrics import classification_report

import math
from argparse import ArgumentParser
import os.path

from train import load_test_indexes
from train_top import load_labels


class FeatureExampleGenerator(object):

    def __init__(self, indexes, feature_dir, batch_size):
        self.indexes = indexes
        self.feature_dir = feature_dir
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

        # Advance pointer for next batch
        self.pointer += self.batch_size
        if self.pointer >= len(self.indexes):
            self.pointer = 0

        return example_array


def predict(model_filename, features_dir, test_indexes, all_labels, batch_size, verbose):

    # Convert list of expected labels to one-hot array
    expected_labels = to_categorical(all_labels[test_indexes])

    # Load model from file
    model = load_model(model_filename)

    # Compute predictions for all test examples
    predictions = model.predict_generator(
        FeatureExampleGenerator(test_indexes, features_dir, batch_size),
        steps=math.ceil(float(len(test_indexes)) / batch_size),
        verbose=(1 if verbose else 0),
    )

    # Predictions are scores for each class, for each example
    # We convert them into a one-hot matrix of labels as follows:
    # 1. Identify all prediction scores below .5.  We mark these as
    #    incapable of resulting in a classifcation as that class
    below_threshold_mask = np.where(predictions < .5)

    # 2. Identify the position of all predictions that aren't the
    #    max in their row.  These also can't be the predicted label.
    not_max_mask = np.ones(predictions.shape, dtype=np.bool_)
    not_max_mask[range(predictions.shape[0]), predictions.argmax(axis=1)] = 0

    # 3. Make one-hot matrix with a one at all locations where the
    #    score is greater than the threshold and the max
    labels = np.ones(predictions.shape, dtype=np.uint8)
    labels[not_max_mask] = 0
    labels[below_threshold_mask] = 0

    print(classification_report(expected_labels, labels))



if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("model", help="H5 for previously trained " +
        "model of the neural network.")
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
    args = argument_parser.parse_args()

    test_indexes = load_test_indexes(args.test_index_file)
    labels = load_labels(args.csvfile)
    predict(
        model_filename=args.model,
        features_dir=args.features_dir,
        test_indexes=test_indexes,
        all_labels=labels,
        batch_size=args.batch_size,
        verbose=args.v,
    )
