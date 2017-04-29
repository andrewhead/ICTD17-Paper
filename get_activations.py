import numpy as np
import argparse
import os.path
import gc
from tqdm import tqdm


def get_activations(features_dir, feature_batch_size,
        exemplar_count, output_filename):

    output_file = open(output_filename, 'w')

    # Get number of examples and shape of features
    num_examples = len(os.listdir(features_dir))
    features_filename = lambda i: os.path.join(features_dir, str(i) + ".npz")
    features_instance = np.load(features_filename(0))["data"]

    # Number of features to load at a time
    feature_index = 0
    while feature_index < len(features_instance):
        num_cols = min(feature_batch_size, len(features_instance) - feature_index)
        example_features = np.zeros((num_examples, num_cols))
        print("Computing activations for features %d through %d..." %
            (feature_index, feature_index + num_cols - 1))

        # Load in this batch of features for all examples
        for i in tqdm(range(num_examples), desc="Loading features"):
            features = np.load(features_filename(i))["data"]
            feature_batch = features[feature_index:feature_index + feature_batch_size]
            example_features[i] = feature_batch

        print("Writing exemplars to file...", end="")
        for fi, example_ranks in enumerate(example_features.argsort(axis=0).T):
            exemplars = example_ranks[::-1][:exemplar_count]
            output_file.write("%d: %s\n" % (fi, exemplars))
        print("done.")

        # Do garbage collection between each batch to make sure
        # we have enough memory for the next one.
        gc.collect()
        feature_index += feature_batch_size

    output_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
        "Get indexes of images that most activate individual " +
        "features in a flattened array of features")
    parser.add_argument("features_dir", help="directory containing " +
        "features processed for each image.")
    parser.add_argument("output", help="file to output the results")
    parser.add_argument("--feature-batch-size", default=10000, type=int,
        help="How many features to load from files at a time.  " +
             "This will vary based on how many examples you have.  " +
             "Do as many as you can without running out of memory.")
    parser.add_argument("--exemplar-count", default=10, type=int,
        help="How many exemplars of each feature to save")

    args = parser.parse_args()
    get_activations(
        features_dir=args.features_dir,
        feature_batch_size=args.feature_batch_size,
        exemplar_count=args.exemplar_count,
        output_filename=args.output,
    )
