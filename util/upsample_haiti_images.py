from util.load_data import load_labels

from argparse import ArgumentParser
import math
import numpy as np
import os
import shutil


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Upsample images for Haiti")
    argument_parser.add_argument("images_dir")
    argument_parser.add_argument("csvfile", help="CSV containing labels")
    argument_parser.add_argument("output_dir", help="Directory to output " +
        "upsampled images")
    argument_parser.add_argument("validation_ratio", help="Proportion of images " +
        "to put in validation set (stratified)", type=float)
    args = argument_parser.parse_args()

    labels = load_labels(args.csvfile)

    # We assume that the largest category will be 0's
    class0_count = len([_ for _ in filter(lambda l: l == '0', labels)])
    training_sample_size = math.floor(class0_count * (1 - args.validation_ratio))
    validation_sample_size = class0_count - training_sample_size
    print("Upsampling to", training_sample_size, "training images")

    example_indexes = []
    for feature_filename in os.listdir(args.images_dir):
        index = int(os.path.splitext(feature_filename)[0])
        example_indexes.append(index)

    upsampled_examples = {'training': {}, 'validation': {}}

    # Sort examples by class
    # XXX: The below 20 lines are copied from `sample.py`
    class_examples = {}
    for example_index in example_indexes:
        class_ = labels[example_index]
        if class_ not in class_examples:
            class_examples[class_] = []
        class_examples[class_].append(example_index)

    def upsample(list_, quota):
        # Repeat the array as many times as it will take to get
        # enough examples for the sample.  Stack a bunch of shuffled
        # repeats on top of each other.  This sampling method lets us
        # avoid repeat sampling until all items have been sampled once.
        examples_array = np.array(examples, dtype=np.int32)
        repeats = math.ceil(float(quota) / len(examples))
        repeated_examples = np.array([], dtype=np.int32)
        for _ in range(repeats):
            repeat = examples_array.copy()
            np.random.shuffle(repeat)
            repeated_examples = np.concatenate((repeated_examples, repeat))
        return repeated_examples[:quota]

    # For each class...
    for class_, examples in class_examples.items():

        split_index = math.floor(len(examples) * (1 - args.validation_ratio))
        validation_examples = examples[split_index:]
        training_examples = examples[:split_index]

        # Truncate the repeated randomized lists to the sample size
        # and append to the shared list of output examples
        upsampled_examples['validation'][class_] =\
             upsample(validation_examples, validation_sample_size)
        upsampled_examples['training'][class_] =\
             upsample(training_examples, training_sample_size)

    # Filter to the indexes that can be used for training
    for example_set, label_data in upsampled_examples.items():
        for label, indexes in label_data.items():
            print("\nMoving images for label", label, ", set", example_set)
            for i, example_index in enumerate(indexes):
                input_name = os.path.join(args.images_dir, str(example_index) + ".jpg")
                output_name = os.path.join(
                    args.output_dir, example_set, label,
                    "original_index_%d_new_index_%d.jpg" % (example_index, i)
                )
                shutil.copyfile(input_name, output_name)
                print(".", end="")
                if (i + 1) % 100 == 0:
                    print("(", i + 1, ")")
