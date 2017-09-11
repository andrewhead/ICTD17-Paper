""" REUSE: A substantial amount of the code in this file was reused from
Problem Set 1 teacher-provided boilerplate and our code.  There's a lot
of logic for clustering images that we didn't want to invent twice. """

import argparse
import numpy as np
import os.path
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.externals import joblib

from util.geometry import MapGeometry
from util.load_data import read_wealth_records, read_education_records,\
    read_water_records, get_map_from_i_j_to_example_index


def get_features_for_clusters(records, features_dir, i_j_to_example_index_map, map_geometry):
    # Returns a numpy array, where each row corresponds to one of the entries
    # in `wealth_records`.  Each row contains the average of the features for
    # all images in that record's cluster.
    # Also returns a list of all clusters for which *no* images were found
    # (may be those right on the border).  The prediction data for these ones
    # should probably be discarded.

    avg_feature_arrays = tuple()
    missing_records = {}
    records_without_any_images = []
    for record_index, record in tqdm(enumerate(records), 
            desc="Loading features for records", total=len(records)):
        
        # Find the neighborhood of images for this record's location
        # Latitude and longitude are more precise, so if they're available, use
        # them for finding the closest set of images in the neighborhood
        if 'longitude' in record and 'latitude' in record:
            neighborhood = map_geometry.get_image_rect_from_long_lat(
                record['longitude'], record['latitude'])
        else:
            neighborhood = map_geometry.get_image_rect_from_cell_indexes(
                record['i'], record['j'])
            centroid_longitude, centroid_latitude = (
                map_geometry.get_centroid_long_lat(record['i'], record['j']))
            # Save references to tthe approximate latitude and longitude,
            # in case we want to use it for printing out debugging info later.
            record['longitude'] = centroid_longitude
            record['latitude'] = centroid_latitude
        
        # Collect features for all images in the neighborhood
        feature_arrays = tuple()
        count_missing = 0
        for image_i in range(neighborhood['left'], neighborhood['left'] + neighborhood['width']):
            for image_j in range(neighborhood['top'], neighborhood['top'] + neighborhood['height']):
                if (image_i, image_j) not in i_j_to_example_index_map:
                    count_missing += 1
                    continue
                example_index = i_j_to_example_index_map[(image_i, image_j)]
                example_features = np.load(os.path.join(
                    features_dir, str(example_index) + ".npz"))["data"]
                feature_arrays += (example_features,)

        # Compute the average of all features over all neighbors
        if len(feature_arrays) > 0:
            cluster_features = np.stack(feature_arrays)
            avg_feature_arrays += (np.average(cluster_features, axis=0),)
                
        if count_missing > 0:
            missing_records[record_index] = count_missing
            if len(feature_arrays) == 0:
                records_without_any_images.append(record_index)

    if len(missing_records.keys()) > 0:
        print("Missing images for %d clusters. " % (len(missing_records.keys())) +
            ". This might not be a bad thing as some clusters may be near a " +
            "border.  These clusters are:")
        for record_index, missing_count in missing_records.items():
            print("Record %d (%f, %f): %d images" % 
                (record_index, records[record_index]['latitude'],
                 records[record_index]['longitude'], missing_count))

    avg_features = np.stack(avg_feature_arrays)
    return avg_features, records_without_any_images


def predict(features, y, Xtest, ytest, output_filename):
    # This method assumes you have already split the data into
    # test data and training data, and are only passing in training data.
    features = np.array(features)
    y = np.array(y)
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    from sklearn.model_selection import KFold

    for i, (train_index, val_index) in enumerate(KFold(n_splits=5).split(features)):

        print("On fold", i)
        Xtrain = features[train_index]
        ytrain = y[train_index]
        Xval = features[val_index]
        yval = y[val_index]

        # Discover best regularsization parameter for this fold
        best_score = -100
        best_alpha = -1
        for alpha in np.logspace(-3, 5, 50, base=10):
            ridge = Ridge(alpha=alpha)
            ridge.fit(Xtrain, ytrain)
            score = ridge.score(Xval, yval)
            if score > best_score:
                best_alpha = alpha
                best_score = score
        print("Stage I best score", best_score, "for alpha", best_alpha)

        tuned_score = -100
        tuned_alpha = -1
        for alpha in np.logspace(np.log10(best_alpha / 2), np.log10(best_alpha * 2), 50, base=10):
            ridge = Ridge(alpha=alpha)
            ridge.fit(Xtrain, ytrain)
            score = ridge.score(Xval, yval)
            if score > tuned_score:
                tuned_alpha = alpha
                tuned_score = score
        print("Stage II best score", tuned_score, "for alpha", tuned_alpha)

        # Run on the hold-out test set
        ridge = Ridge(alpha=tuned_alpha)
        ridge.fit(features, y)
        print("Test best score:", ridge.score(Xtest, ytest))

    # Retrain the model on all training data, and dump it to a file for later
    print("Saving trained model to file ", output_filename)
    joblib.dump(ridge, output_filename)

    return ridge


def train_development(features_dir, wealth_csv, education_csv, water_csv,
    nightlights_csv, nightlights_raster, output_basename, v):

    if v:
        print("Loading map geometry...", end="")
    map_geometry = MapGeometry(nightlights_raster)
    if v:
        print(".")
    i_j_to_example_index_map = get_map_from_i_j_to_example_index(nightlights_csv)
 
    # Predict wealth
    if v:
        print("Preparing for wealth predictions.")
    wealth_records = read_wealth_records(args.wealth_csv)
    y_wealth = [r['wealth'] for r in wealth_records]
    X_wealth, records_to_discard = get_features_for_clusters(
        records=wealth_records,
        features_dir=args.features_dir,
        i_j_to_example_index_map=i_j_to_example_index_map,
        map_geometry=map_geometry,
    )
    # Some of the clusters might not have any images.  Just discard the
    # prediction for these ones, don't factor it into the model.  Make
    # sure to discard in reverse, so we don't mess up the indexing
    # for discarding later records after discarding earlier records.
    for i in reversed(records_to_discard):
        del(y_wealth[i])
    X_wealth_train, X_wealth_test, y_wealth_train, y_wealth_test = (
        train_test_split(X_wealth, y_wealth, test_size=0.25, random_state=None))
    print("Now predicting wealth...")
    wealth_model = predict(X_wealth_train, y_wealth_train,
        X_wealth_test, y_wealth_test,
        args.output_basename + "_wealth.pkl")

    # Predict education
    if v:
        print("Preparing for education predictions.")
    education_records = read_education_records(education_csv)
    y_education = [r['education_index'] for r in education_records]
    X_education, records_to_discard = get_features_for_clusters(
        records=education_records,
        features_dir=features_dir,
        i_j_to_example_index_map=i_j_to_example_index_map,
        map_geometry=map_geometry,
    )
    for i in reversed(records_to_discard):
        del(y_education[i])
    X_education_train, X_education_test, y_education_train, y_education_test = (
        train_test_split(X_education, y_education, test_size=0.25, random_state=None))
    print("Now predicting education...")
    education_model = predict(X_education_train, y_education_train,
        X_education_test, y_education_test,
        args.output_basename + "_education.pkl")
        
    # Predict Water
    if v:
        print("Preparing for water predictions.")
    water_records = read_water_records(water_csv)
    y_water = [r['water_index'] for r in water_records]
    X_water, records_to_discard = get_features_for_clusters(
        records=water_records,
        features_dir=features_dir,
        i_j_to_example_index_map=i_j_to_example_index_map,
        map_geometry=map_geometry,
    )
    for i in reversed(records_to_discard):
        del(y_water[i])
    X_water_train, X_water_test, y_water_train, y_water_test = (
        train_test_split(X_water, y_water, test_size=0.25, random_state=None))
    print("Now predicting water...")
    water_model = predict(X_water_train, y_water_train,
        X_water_test, y_water_test,
        args.output_basename + "_water.pkl")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train models to predict " +
        "wealth, water and education indexes from arbitrary features")
    parser.add_argument("features_dir", help="directory containing features, " +
        "with one file containing a flat numpy array (.npz) per image")
    parser.add_argument("wealth_csv", help="CSV file where " +
        "the top row is a header, col 1 (zero-indexed) is the wealth index, " +
        "col 7 is the latitude, and col 8 is the longitude.")
    parser.add_argument("education_csv", help="CSV file where " +
        "the top row is a header, col 3 (zero-indexed) is the education index, " +
        "col 1 is the cell's 'i' coordinate, and col 2 is the 'j' coordinate.")
    parser.add_argument("water_csv", help="CSV file where " +
        "the top row is a header, col 4 (zero-indexed) is the water index, " +
        "col 2 is the cell's 'i' coordinate, and col 3 is the 'j' coordinate.")
    parser.add_argument("nightlights_csv", help="CSV file where " +
        "the top row is a header, col 0 (zero-indexed) is the index of the " +
        "example (basename of eature file), and cols 2 and 3 are the " +
        "i and j of the cell in the nightlights data")
    parser.add_argument("nightlights_raster", help="Raster file of " +
        "nightlights, used for making a map from latitude and longitude " +
        "to cell indexes on the map.")
    parser.add_argument("--prediction-output-basename",
        help="If you're running test results, set this flag and you can " +
        "output the test predictions to file")
    parser.add_argument("output_basename", help="Basename of files to which " +
        "to output the created models.")
    parser.add_argument("-v", action="store_true", help="verbose")
    args = parser.parse_args()

    train_development(
        args.features_dir,
        args.wealth_csv,
        args.education_csv,
        args.water_csv,
        args.nightlights_csv,
        args.nightlights_raster,
        args.output_basename,
        args.v,
    )
