"""
Given a pickled model and a set of examples in an `npz` file (already split
into `X_train` and `X_test`, output predicted and actual results.  Print
out one line for each example, with the first column containing the
actual output and the second containing the predicted output.
"""

import argparse
import numpy as np
from sklearn.externals import joblib


def get_expected_values(y_train, y_test):
    """ Concatenate `y` data into single list of expected y values """
    y_all = np.concatenate((y_train, y_test))
    return y_all


def get_predictions(model, x_train, x_test):
    """ Predict `y` values for all input X """
    x_all = np.concatenate((x_train, x_test))
    y_predicted = model.predict(x_all)
    return y_predicted


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Output expected and predicted values for a set of " +
        "examples and a model")
    parser.add_argument("model", help="pkl file containing model")
    parser.add_argument(
        "data",
        help=".npz file. Expected to contain the following fields: " +
        "X_train, X_test, y_train, y_test")
    parser.add_argument(
        "--report-r2",
        action="store_true",
        help="Whether to print out R^2 value before expected and " +
        "predicted values")
    args = parser.parse_args()

    model_loaded = joblib.load(args.model)
    data = np.load(args.data)
    X_train_loaded = data['X_train']
    X_test_loaded = data['X_test']
    y_train_loaded = data['y_train']
    y_test_loaded = data['y_test']

    y_expected_results = get_expected_values(y_train_loaded, y_test_loaded)
    y_predicted_results = get_predictions(model_loaded, X_train_loaded, X_test_loaded)

    if args.report_r2:
        x_all_loaded = np.concatenate((X_train_loaded, X_test_loaded))
        print("R^2:", model_loaded.score(x_all_loaded, y_expected_results))

    print("Example Index,Expected,Predicted")
    for i in range(len(y_expected_results)):
        print(",".join([str(_) for _ in
                        [i, y_expected_results[i], y_predicted_results[i]]]))
