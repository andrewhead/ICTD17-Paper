import numpy as np
import argparse
import os.path
import gc
from tqdm import tqdm


def get_activations(features_dir, exemplar_count, output_filename):

    output_file = open(output_filename, 'w')

    # Get number of examples and shape of features
    num_examples = len(os.listdir(features_dir))
    features_filename = lambda i: os.path.join(features_dir, str(i) + ".npz")
    features_instance = np.load(features_filename(0))["data"]

    # Find out how many filters we'll be averaging over, and the
    # axes of the features over which to average to get the average intensity
    # within each filter as the filter "activations"
    features_shape = features_instance.shape
    num_filters = features_shape[-1]
    within_filter_axes = tuple(range(len(features_shape) - 1))

    # Make an array of zeros, with one row for each example and one
    # and one column for the "activation" in each filter
    filter_activations = np.zeros((num_examples, num_filters))

    # Load in this batch of features for all examples
    for i in tqdm(range(num_examples), desc="Loading features"):
        features = np.load(features_filename(i))["data"]
        example_filter_averages = np.average(features, axis=within_filter_axes)
        filter_activations[i] = example_filter_averages

    print("Writing exemplars to file...", end="")
    for fi, example_ranks in enumerate(filter_activations.argsort(axis=0).T):
        exemplars = example_ranks[::-1][:exemplar_count]
        output_file.write("%d: %s\n" % (fi, exemplars))
    print("done.")

    output_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
        "Get indexes of images that most activate individual " +
        "filters in a set of provided features")
    parser.add_argument("features_dir", help="directory containing " +
        "features processed for each image.")
    parser.add_argument("output", help="file to output the results")
    parser.add_argument("--exemplar-count", default=10, type=int,
        help="How many exemplars of each feature to save")

    args = parser.parse_args()
    get_activations(
        features_dir=args.features_dir,
        exemplar_count=args.exemplar_count,
        output_filename=args.output,
    )
