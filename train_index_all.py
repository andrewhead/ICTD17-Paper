""" REUSE: A substantial amount of the code in this file was reused from
Problem Set 1 teacher-provided boilerplate and our code.  There's a lot
of logic for clustering images that we didn't want to invent twice. """

import argparse
import numpy as np
import os.path
from tqdm import tqdm

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.externals import joblib

from util.geometry import MapGeometry
from util.load_data import read_records, get_map_from_i_j_to_example_index


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


def predict(features, y): #, Xtest, ytest):
    # This method assumes you have already split the data into
    # test data and training data, and are only passing in training data.
    features = np.array(features)
    y = np.array(y)
    # Xtest = np.array(Xtest)
    # ytest = np.array(ytest)

    from sklearn.model_selection import KFold

    best_alpha_overall = -1
    best_score_overall = -1
    for i, (train_index, val_index) in enumerate(KFold(n_splits=5, shuffle=True, random_state=443352346).split(features)):

        print("On fold", i)
        Xtrain = features[train_index]
        ytrain = y[train_index]
        Xval = features[val_index]
        yval = y[val_index]

        # Discover best regularization parameter for this fold
        model = RidgeCV(alphas=np.logspace(-3, 5, 50, base=10), cv=5)
        model.fit(Xtrain, ytrain)
        best_alpha = model.alpha_
        print("Stage I best alpha", best_alpha)

        model = RidgeCV(alphas=np.logspace(np.log10(best_alpha / 2), np.log10(best_alpha * 2), 50, base=10), cv=5)
        model.fit(Xtrain, ytrain)
        tuned_alpha = model.alpha_
        print("Stage II best alpha", tuned_alpha)

        # Run on the hold-out test set
        ridge = Ridge(alpha=tuned_alpha)
        ridge.fit(Xtrain, ytrain)
        # score = ridge.score(Xval, yval)
        print("Test best score:", ridge.score(Xval, yval))

    # Retrain the model on all training data, and dump it to a file for later
    # print("Saving trained model to file ", output_filename)
    # joblib.dump(ridge, output_filename)

    return ridge


def get_random_seeds():
    # Make sure that every time we get these seeds for splitting, they
    # have the same values.
    np.random.seed(443352346)
    return np.random.randint(0, 2**32 - 1, size=10).tolist()


def do_predictions_for(features_dir, csv, metric_name, metric_column,
    i_j_to_example_index_map, map_geometry, v): 

    if v:
        print("Preparing for", metric_name, "predictions.")

    try:
        records = read_records(csv, metric_column)
    except Exception as e:
        print(e)
        print("Error: couldn't load CSV file", csv, ", skipping metric.")
        return

    y = [float(r[metric_column]) for r in records]
    X, records_to_discard = get_features_for_clusters(
        records=records,
        features_dir=features_dir,
        i_j_to_example_index_map=i_j_to_example_index_map,
        map_geometry=map_geometry,
    )

    # Some of the clusters might not have any images.  Just discard the
    # prediction for these ones, don't factor it into the model.  Make
    # sure to discard in reverse, so we don't mess up the indexing
    # for discarding later records after discarding earlier records.
    for i in reversed(records_to_discard):
        del(y[i])

    # for split_seed in get_random_seeds():
    # X_train, X_test, y_train, y_test = (
    #     train_test_split(X, y, test_size=0.25, random_state=split_seed))
    print("Now predicting", metric_name, "...")
    wealth_model = predict(X, y)


def train_development(features_dir, country_name, nightlights_csv, nightlights_raster, v):

    if v:
        print("Loading map geometry...", end="")
    map_geometry = MapGeometry(nightlights_raster)
    if v:
        print(".")
    i_j_to_example_index_map = get_map_from_i_j_to_example_index(nightlights_csv)
 
    csv_file = lambda basename: os.path.join("csv", country_name + "_" + basename + ".csv")

    def run_predictions(csv, metric_name, metric_column):
        do_predictions_for(features_dir, csv, metric_name, metric_column,
            i_j_to_example_index_map, map_geometry, v)

    # run_predictions(csv_file("DHS_wealth"), "wealth", "wealth")
    # run_predictions(csv_file("cluster_avg_educ_nightlights"), "education", "avg_educ_index")
    run_predictions(csv_file("cluster_avg_water_nightlights"), "water", "water")
    # run_predictions(csv_file("height_4_age"), "child height percentile", "height_4_age")
    # run_predictions(csv_file("weight_4_age"), "child weight percentile", "weight_4_age")
    # run_predictions(csv_file("weight_4_height"), "child weight / height percentile", "weight_4_height")
    # run_predictions(csv_file("female_bmi"), "female BMI", "female_bmi")
    # run_predictions(csv_file("bed_net_num"), "bed net count", "bed_net_num")
    # run_predictions(csv_file("hemoglobin"), "hemoglobin level", "hemoglobin")
    # run_predictions(csv_file("electricity"), "electricity", "electricity")
    # run_predictions(csv_file("mobile"), "mobile phone ownership", "mobile")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train models to predict " +
        "wealth, water and education indexes from arbitrary features")
    parser.add_argument("features_dir", help="directory containing features, " +
        "with one file containing a flat numpy array (.npz) per image")
    parser.add_argument("country_name", help="lower case")
    parser.add_argument("nightlights_csv", help="CSV file where " +
        "the top row is a header, col 0 (zero-indexed) is the index of the " +
        "example (basename of eature file), and cols 2 and 3 are the " +
        "i and j of the cell in the nightlights data")
    parser.add_argument("nightlights_raster", help="Raster file of " +
        "nightlights, used for making a map from latitude and longitude " +
        "to cell indexes on the map.")
    parser.add_argument("-v", action="store_true", help="verbose")
    args = parser.parse_args()

    train_development(
        args.features_dir,
        args.country_name,
        args.nightlights_csv,
        args.nightlights_raster,
        args.v,
    )
