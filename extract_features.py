from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.utils.generic_utils import get_custom_objects
from keras_models.vgg16 import VGG16
import numpy as np

import math
import os.path
import argparse
from tqdm import tqdm


def get_features(model, img_paths, batch_size):

    # Load first image to get image dimensions
    img = image.load_img(img_paths[0])
    width = img.width
    height = img.height

    batch_start = 0
    while batch_start < len(img_paths):

        num_rows = min(batch_size, len(img_paths) - batch_start)
        # XXX: I'm not sure if the order of height and width is correct here,
        # but it doesn't matter for us right now as we're using square images
        X = np.zeros((num_rows, width, height, 3))
        batch_img_paths = img_paths[batch_start:batch_start + batch_size]

        for img_index, img_path in enumerate(batch_img_paths):

            # Make each image a single 'row' of a tensor
            img = image.load_img(img_path)
            img_array = image.img_to_array(img)
            X[img_index, :, :, :] = img_array

        # Find the output of the final layer
        # Borrowed from https://github.com/fchollet/keras/issues/41
        features = model.predict([X])
        features_array = np.array(features)
        yield features_array
        batch_start += batch_size

    yield StopIteration


def extract_features(model_path, image_dir, layer_name, output_dir,
        flatten, batch_size):

    # Add records for the custom metrics we attached to the models,
    # pointing them to no-op metrics methods.
    custom_objects = get_custom_objects()
    custom_objects.update({"recall (C0)": lambda x, y: K.constant(0)})
    custom_objects.update({"% examples (C0)": lambda x, y: K.constant(0)})
    custom_objects.update({"recall (C1)": lambda x, y: K.constant(0)})
    custom_objects.update({"% examples (C1)": lambda x, y: K.constant(0)})
    custom_objects.update({"recall (C2)": lambda x, y: K.constant(0)})
    custom_objects.update({"% examples (C2)": lambda x, y: K.constant(0)})

    # Create a model that only outputs the requested layer
    if model_path:
        print("Loading model %s..." % (model_path), end="")
        model = load_model(model_path)
    else:
        print("Loading VGG16...", end="")
        model = VGG16(weights='imagenet', include_top=False)
    print("done.")
    print("Adjusting model to output feature layer..." , end="")
    feature_layer = model.get_layer(layer_name)
    model = Model(model.input, feature_layer.output)
    print("done.")
    
    # Collect the paths of all of the images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    # Compute the features in batches
    print("Now computing features for all batches of images...")
    expected_batches = math.ceil(len(image_paths) / float(batch_size))
    filename = lambda image_index: os.path.join(
        output_dir, str(image_index) + ".npz")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_index = 0
    for feature_batch in tqdm(
            get_features(model, image_paths, batch_size),
            total=expected_batches):
        if flatten:
            feature_batch = feature_batch.reshape(feature_batch.shape[0], -1)
        for image_features in feature_batch:
            # It's important to store using `compressed` if you want to save more than
            # a few hundred images.  Without compression, every 1,000 images will take
            # about 1GB of memory, which might not scale well for most datasets
            # Each record is saved to its own file to enable efficient loading without
            # needing to load all image features into memory during later training.
            np.savez_compressed(filename(image_index), data=image_features)
            image_index += 1

    print("All features have been computed and saved.")
    print("Reload features for each image with: `np.load(<filename>)['data']`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        "Extract features for a set of images from a layer of " +
        "a neural network model."
        )
    parser.add_argument("image_dir", help="Directory containing all images")
    parser.add_argument("layer_name", help="Layer from which to extract features")
    parser.add_argument("--model", help="H5 file containing model.  If not " +
        "provided, then features are extracted using VGG16")
    parser.add_argument("--flatten", action="store_true", help=
        "Whether to flatten the extracted features.  This is useful if you " +
        "want to train regression using the extracted features."
        )
    parser.add_argument("--batch-size", default=32, help="Number of images to " + 
        "extract features for at a time.", type=int)
    parser.add_argument("--output-dir", default="features.npz",
        help="Name of file to write features to.")

    args = parser.parse_args()
    extract_features(
        args.model, args.image_dir, args.layer_name, args.output_dir, 
        args.flatten, args.batch_size)
