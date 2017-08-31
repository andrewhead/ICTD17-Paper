from argparse import ArgumentParser
import os.path

from tune_block5 import train
from extract_features import extract_features
from train_index_all import train_development
from util.load_data import load_labels


BLOCK5_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Train top layers of neural net")
    argument_parser.add_argument("features_dir")
    argument_parser.add_argument("top_model", help="H5 for previously trained " +
        "top layers of the neural network.")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument(
        "-v", action="store_true",
        help="Print out detailed info about progress.")
    argument_parser.add_argument("--training-indexes-file", help="File containing " +
        "an index of a training example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    argument_parser.add_argument("--validation-indexes-file", help="File containing " +
        "an index of a validation example on each line.  Useful if you only have " +
        "features extracted for a subset of the examples.")
    args = argument_parser.parse_args()

    # Load in the class labels
    labels = load_labels(args.csvfile)

    # Load indexes of examples that will be used in the second step of transfer learning
    cluster_example_indexes = []
    with open(os.path.join("indexes", "Haiti_dhs_cluster_indexes")) as indexes_file:
        for line in indexes_file:
            cluster_example_indexes.add(int(line.strip()))

    # For each hyper-parameter setting...
    for learning_rate in [.1, .5, .25, .01, .05, .025, .001, .005, .0025, .0001]:
        for batch_size in [8, 16, 32, 64, 128, 256]:
            print("Fine tuning with learning rate", learning_rate, "and batch size", batch_size)
            try:
                # Fine-tune the model
                final_model_path = train(
                    args.features_dir,
                    args.top_model,
                    args.labels,
                    batch_size,
                    learning_rate,
                    0.9,
                    50,
                    args.training_indexes_file,
                    args.validation_indexes_file,
                    True
                )
            except Exception as e:
                # If something went wrong, just skip these hyperparameters.
                print("Some exception occurred", e)
                print("This was for hyperparameters lr", learning_rate, "batch size", batch_size)
                print("Moving on to the next one")
                continue
    
            # Extract features from the final layer (to use for predictions)
            print("Extracting final layer features for model", final_model_path)
            output_features_dir = os.path.join("features", "haiti_revamped_vgg16_conv7_flattened_" + str(batch_size) + "_" + str(learning_rate)),
            extract_features(
                model_path=final_model_path,
                input_dir=os.path.join("features", "haiti_revamped_vgg16_block4_pool"),
                layer_name="conv7",
                output_dir=output_features_dir,
                flatten=True,
                batch_size=32,
                input_type="features",
                example_indexes=cluster_example_indexes,
            )

            print("Predicting development measures using features in", output_features_dir)
            train_development(
                output_features_dir,
                os.path.join("csv", "haiti_DHS_wealth.csv"),
                os.path.join("csv", "haiti_cluster_avg_educ_nightlights.csv"),
                os.path.join("csv", "haiti_cluster_avg_water_nightlights.csv"),
                os.path.join("csv", "haiti_TL.csv"),
                os.path.join("nightlights", "F182010.v4d_web.stable_lights.avg_vis.tif"),
                os.path.join("models", "indexes", "haiti_revamped_vgg16_tuned_" + str(batch_size) + "_" + str(learning_rate)),
                True,
            )
