import numpy as np
import argparse
import os.path
import gc
from tqdm import tqdm


def get_activations(features_dir, exemplar_count, output_filename):

    output_file = open(output_filename, 'w')

    # Get number of examples and shape of features
    num_examples = len(os.listdir(features_dir))
    features_instance = np.load(os.path.join(features_dir, os.listdir(features_dir)[0]))["data"]
    features_shape = features_instance.shape

    # Find out how many filters we'll be averaging over, and the
    # axes of the features over which to average to get the average intensity
    # within each filter as the filter "activations"
    num_filters = features_shape[-1]
    within_filter_axes = tuple(range(len(features_shape) - 1))

    # Make an array of zeros, with one row for each example and one
    # and one column for the "activation" in each filter
    filter_activations = np.zeros((num_examples, num_filters))

    # Load in this batch of features for all examples
    example_indexes = []
    for i, filename in tqdm(enumerate(os.listdir(features_dir)),
            total=num_examples, desc="Loading features"):

        # Store a link from the example's index in the numpy array
        # and the index of the example in the features directory.       
        example_index = int(os.path.splitext(filename)[0])
        example_indexes.append(example_index)

        # Load the features for this example
        path = os.path.join(features_dir, filename)
        features = np.load(path)["data"]

        # Compute the activations for the example's features
        example_filter_averages = np.average(features, axis=within_filter_axes)
        filter_activations[i] = example_filter_averages

    # Iterate through each filter, with a sorted list of which rows maximize each one.
    # Remember that these row indexes need to be mapped back to example indexes
    # in the original feature directory.
    print("Writing exemplars to file...", end="")
    for filter_index, example_ranks in enumerate(filter_activations.argsort(axis=0).T):

        # Extract the top N exemplars that maximize each filter
        exemplar_rows = example_ranks[::-1][:exemplar_count]

        # Find out the example indexes for each row in the maximizing rows
        exemplars = [example_indexes[row_index] for row_index in exemplar_rows]

        # Write list of exemplars to file
        output_file.write("%d: [" % (filter_index,))
        for exemplar in exemplars:
            output_file.write("%d " % (exemplar,))
        output_file.write("]\n")

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
