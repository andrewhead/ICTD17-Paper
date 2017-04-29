""" REUSE: A substantial amount of the code in this file was reused from
Problem Set 1 teacher-provided boilerplate and our code.  There's a lot
of logic for clustering images that we didn't want to invent twice. """

import argparse
from osgeo import gdal, ogr, osr
import numpy as np
import csv
import os.path
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.externals import joblib


class MapGeometry(object):
    """
    This class is an eyesore, but unfortunately it's necessary for finding
    images within the vicinity of a latitude and longitude, so that we can
    predict a `y` using average features over a larger spatial area.
    """
    def __init__(self, raster_filename):
        self.load_raster(raster_filename)

    def load_raster(self, raster_filename):

        raster_dataset = gdal.Open(raster_filename, gdal.GA_ReadOnly)

        # get project coordination
        proj = raster_dataset.GetProjectionRef()
        bands_data = []

        # Loop through all raster bands
        for b in range(1, raster_dataset.RasterCount + 1):
            band = raster_dataset.GetRasterBand(b)
            bands_data.append(band.ReadAsArray())
            no_data_value = band.GetNoDataValue()
        bands_data = np.dstack(bands_data)
        rows, cols, n_bands = bands_data.shape

        # Get the metadata of the raster
        geo_transform = raster_dataset.GetGeoTransform()
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = geo_transform
        
        # Get location of each pixel
        x_size = 1.0 / int(round(1 / float(x_size)))
        y_size = - x_size
        y_index = np.arange(bands_data.shape[0])
        x_index = np.arange(bands_data.shape[1])
        top_left_x_coords = upper_left_x + x_index * x_size
        top_left_y_coords = upper_left_y + y_index * y_size

        # Add half of the cell size to get the centroid of the cell
        centroid_x_coords = top_left_x_coords + (x_size / 2)
        centroid_y_coords = top_left_y_coords + (y_size / 2)

        self.x_size = x_size
        self.top_left_x_coords = top_left_x_coords
        self.top_left_y_coords = top_left_y_coords
        self.centroid_x_coords = centroid_x_coords
        self.centroid_y_coords = centroid_y_coords

    def get_cell_idx(self, lon, lat):
        lon_idx = np.where(self.top_left_x_coords < lon)[0][-1]
        lat_idx = np.where(self.top_left_y_coords > lat)[0][-1]
        return lon_idx, lat_idx

    def get_image_rect(self, longitude, latitude):
        """
        We want to get a 10x10 matrix of images around this image. All we have is the 
        center cell indexes and latitude and longitude of the center. We can't just 
        expand to 5 on either side, as this will give us an 11x11 matrix of images.
        So we can create this 11x11 matrix, and truncate whichever sides are
        farthest away from the center latitude and longitude.

        In practice, I compute this as follows. I pick the image 5 to the left of
        the center image (the left image). Then I compute the longitude of the
        ideal left boundary of the matrix, if the center coordinates were right in 
        the middle of the 10x10 cell. If the ideal left boundary is closer to right 
        side of the left image than the left side, then less than half of this 
        column of images would fit within the ideal image matrix: we truncate the
        left side. Otherwise, less than half of the right column would fit in the
        ideal image matrix; we truncate the right side. We use the same logic to
        decide whether to truncate the top or bottom of the 11x11 matrix.
        """
        (image_i, image_j) = self.get_cell_idx(longitude, latitude)
        
        left_image_i = image_i - 5
        left_image_center_longitude = self.centroid_x_coords[left_image_i]
        ideal_left_longitude =  longitude - self.x_size * 5
        truncate_left = (ideal_left_longitude > left_image_center_longitude)
                    
        top_image_j = image_j - 5
        top_image_center_latitude = self.centroid_y_coords[top_image_j]
        ideal_top_latitude = latitude + self.x_size * 5  # (latitude gets more positive as we go up)
        truncate_top = (ideal_top_latitude < top_image_center_latitude)
        
        rect = {'width': 10, 'height': 10}
        rect['left'] = image_i - 4 if truncate_left else image_i - 5
        rect['top'] = image_j - 4 if truncate_top else image_j - 5
        return rect


def get_features_for_clusters(records, features_dir, i_j_to_example_index_map, map_geometry):
    # Returns a numpy array, where each row corresponds to one of the entries
    # in `wealth_records`.  Each row contains the average of the features for
    # all images in that record's cluster.

    avg_feature_arrays = tuple()
    missing_records = {}
    for record_index, record in tqdm(enumerate(records), 
            desc="Loading features for records", total=len(records)):
        
        # Find the neighborhood of images for this record's location
        neighborhood = map_geometry.get_image_rect(
            record['longitude'], record['latitude'])
        
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
        cluster_features = np.stack(feature_arrays)
        avg_feature_arrays += (np.average(cluster_features, axis=0),)
                
        if count_missing > 0:
            missing_records[record_index] = count_missing

    if len(missing_records.keys()) > 0:
        print("Missing images for %d clusters.  This might not be a bad " +
            "thing: some clusters may be near a border.  These clusters are:" % 
            (len(missing_records.keys())))
        for record_index, missing_count in missing_records.items():
            print("Record %d (%f, %f): %d images" % 
                (record_index, records[record_index]['latitude'],
                 records[record_index]['longitude'], missing_count))

    avg_features = np.stack(avg_feature_arrays)
    return avg_features


def read_wealth_records(csv_path):

    records = []

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Cast longitude, latitude, and wealth to numbers
            row['id'] = int(row['id'])
            row['wealth'] = float(row['wealth'])
            row['latitude'] = float(row['LATNUM'])
            row['longitude'] = float(row['LONGNUM'])
            records.append(row)

    return records


def get_map_from_i_j_to_example_index(nightlights_csv_path):
    # Later on, we're going to have to go from an `i` and `j` of a cell
    # in the raster map to an example index.  We've luckily already
    # stored the relationship between these in a CSV file.  We just have
    # to hydrate it into a map.
    i_j_to_example_dict = {}
    with open(nightlights_csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Cast longitude, latitude, and wealth to numbers
            id_ = int(row['id'])
            i = int(row['full_i'])
            j = int(row['full_j'])
            i_j_to_example_dict[(i, j)] = id_

    return i_j_to_example_dict


def predict(features, y, output_filename):
    # This method assumes you have already split the data into
    # test data and training data, and are only passing in training data.

    # Do cross-validation for this model
    ridge = Ridge()
    r2_values = cross_val_score(ridge, features, y, cv=10, scoring='r2')
    print("Cross-validation results:")
    print("All R^2:", r2_values)
    print("Average R^2:", np.average(r2_values))

    # Retrain the model on all training data, and dump it to a file for later
    ridge = Ridge()
    ridge.fit(features, y)
    print("Saving trained model to file ", output_filename)
    joblib.dump(ridge, output_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train models to predict " +
        "wealth and education indexes from arbitrary features")
    parser.add_argument("features_dir", help="directory containing features, " +
        "with one file containing a flat numpy array (.npz) per image")
    parser.add_argument("wealth_csv", help="CSV file where " +
        "the top row is a header, col 1 (zero-indexed) is the wealth index, " +
        "col 7 is the latitude, and col 8 is the longitude.")
    parser.add_argument("nightlights_csv", help="CSV file where " +
        "the top row is a header, col 0 (zero-indexed) is the index of the " +
        "example (basename of eature file), and cols 2 and 3 are the " +
        "i and j of the cell in the nightlights data")
    parser.add_argument("nightlights_raster", help="Raster file of " +
        "nightlights, used for making a map from latitude and longitude " +
        "to cell indexes on the map.")
    parser.add_argument("output_basename", help="Basename of files to which " +
        "to output the created models.")
    parser.add_argument("-v", action="store_true", help="verbose")
    args = parser.parse_args()

    if args.v:
        print("Loading map geometry...", end="")
    map_geometry = MapGeometry(args.nightlights_raster)
    if args.v:
        print(".")
    i_j_to_example_index_map = get_map_from_i_j_to_example_index(args.nightlights_csv)
    
    # Predict wealth
    wealth_records = read_wealth_records(args.wealth_csv)
    X_wealth = get_features_for_clusters(
        records=wealth_records,
        features_dir=args.features_dir,
        i_j_to_example_index_map=i_j_to_example_index_map,
        map_geometry=map_geometry,
    )
    y_wealth = np.array([r['wealth'] for r in wealth_records])
    X_wealth_train, X_wealth_test, y_wealth_train, y_wealth_test = (
        train_test_split(X_wealth, y_wealth, test_size=0.33, random_state=1))
    print("Now predicting wealth...")
    predict(X_wealth_train, y_wealth_train, args.output_basename + "_wealth.pkl")
    
