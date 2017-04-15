import random
from argparse import ArgumentParser
import os


if __name__ == "__main__":

    argument_parser = ArgumentParser(description="Make list of image indexes for test set")
    argument_parser.add_argument("input_dir", help="Directory containing images")
    argument_parser.add_argument("ratio", type=float, default=0.1,
        help="Proportion of original images to add to test set. " +
             "If images don't divide evenly by the proportion, the number of images " +
             "is rounded down to the nearest whole number.")
    args = argument_parser.parse_args()

    filenames = os.listdir(args.input_dir)
    image_indexes = [int(filename.replace(".jpg", "")) for filename in filenames]

    test_sample_size = int(len(image_indexes) * args.ratio)
    test_image_indexes = random.sample(image_indexes, test_sample_size)
    for index in test_image_indexes:
        print(index)

