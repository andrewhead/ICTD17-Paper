# DiiD-Predictor

You can use the following commands to do training and testing for our DiiD final project.
Before you run any of these commands, do this:

```bash
sudo -u andrew bash       # Change user to Andrew
cd ~/DiiD-Predictor       # Change directory to this directory
source venv/bin/activate  # Start Python virtual environment (load dependencies)
```

Most of these commands will take a *really long* time to run.  To make sure that they
still run even after your SSH session has ended, use `nohup` at the beginning
of every command.  You can start off a command with `nohup` and then watch it's
progress with the following recipe:

```bash
nohup <command> &  # run the command, don't terminate when SSH session is closed
tail -f nohup.log  # view the output of the program
```

For example, to extract features, run:
```bash
nohup python extract_features.py \
  images/Rwanda_simple/ \
  block5_pool \
  --output-dir features/rwanda_vgg16_block5_pool/ &
tail -f nohup.log
```

## Preprocess images to have the expected indexes

*Note*: The `rwanda_TL.csv` file comes will have to be uploaded securely (it's protected data).
Before doing this, create a `csv` directory in the `DiiD-Predictor` directory.
For uploading this file with `scp`, see related instructions for connecting over `ssh` (here)[https://cloud.google.com/compute/docs/instances/connecting-to-instance#standardssh].
Then, run the `copy_images.py` script.

```bash
python copy_images.py \
  images/Rwanda/ \
  csv/rwanda_TL.csv \
  --output-dir images/Rwanda_simple/
```

## Split into the test set

```bash
python util/get_test_indexes.py \
  images/Rwanda_simple/ 0.1 > indexes/Rwanda_test_indexes.txt
```

## Extract features from an arbitrary layer of a neural network model

```bash
python extract_features.py \
  images/Rwanda_simple/ \
  block5_pool \
  --output-dir features/rwanda_vgg16_block5_pool/
```

The `--flatten` flag is optional, and it flattens the feature array for each image.  Replace `"block5_conv3"` with the name of the layer you want to extract features for.  You can also set the number of images to process together with the `--batch-size` argument.  While this extracts features from an ImageNet VGG16 model, you can also provide another model with the `--model` option.

## Train the top layers of the neural network

To train the top layers:
```bash
python train_top.py \
  features/rwanda_vgg16_block5_pool \
  csv/rwanda_TL.csv \
  indexes/Rwanda_test_indexes.txt \
  --learning-rate 0.01 \
  --batch-size=100 \
  --epochs=6 \
  -v
```

Note that this steps depends on having run the previous steps to have the CSV file for Rwanda data, having the images moved into simple indexing form (single integer in the title), and having generated the test set image indexes.

## Get final layer features (can be used for predicting ed and wealth index)

We rely on the `extract_features` script once again:

```bash
python extract_features.py \
  features/rwanda_vgg16_block5_pool \
  conv7 \
  --flatten \
  --model models/rwanda_vgg16_trained_top.h5 \
  --input-type=features \
  --batch-size=16 \
  --output-dir=features/rwanda_vgg16_trained_top_conv7_flattened/
```

## Train a model for predicting wealth and education index

```bash
python train_index.py \
  features/rwanda_vgg16_trained_top_conv7_flattened/ \
  csv/rwanda_DHS_wealth.csv \
  csv/rwanda_cluster_avg_educ_nightlights.csv \
  csv/rwanda_TL.csv \
  nightlights/F182010.v4d_web.stable_lights.avg_vis.tif \
  models/indexes/rwanda_vgg16_trained_top \
  -v
```

Add the `--show-test-results` flag, and optionally set the
`--prediction-output-basename` option, to see and save the
model performance on the test set.

## Train a model for predicting wealth, education and water index
```
python train_index_water.py \
  features/rwanda_vgg16_trained_top_conv7_flattened/ \
  csv/rwanda_DHS_wealth.csv \
  csv/rwanda_cluster_avg_educ_nightlights.csv \
  csv/rwanda_cluster_avg_water_nightlights.csv \
  csv/rwanda_TL.csv \
  nightlights/F182010.v4d_web.stable_lights.avg_vis.tif \
  models/indexes/rwanda_vgg16_trained_top \
  -v
  ```

## Retrain convolutional layers

Extract the features in block 4:

```bash
python extract_features.py \
  images/Rwanda_simple/ \
  block4_pool \
  --output-dir features/rwanda_vgg16_block4_pool/
```

Then retrain only the end of top of the net and the last convolutional block.
(At the time of writing this, this step wasn't yet implemented.)

```bash
python tune_block5.py \
  features/rwanda_vgg16_block4_pool \
  models/rwanda_vgg16_trained_top.h5 \
  csv/rwanda_TL.csv \
  indexes/Rwanda_test_indexes.txt \
  --batch-size=100 \
  --learning-rate=.0001 \
  --epochs=6 \
  -v
```

### And then extract the final layer features...

First, by flattening the learned model:
```bash
python flatten_tuned_model.py \
  models/rwanda_vgg16_tuned.h5 \
  models/rwanda_vgg16_tuned_flattened.h5
```

And then by extracting flattened final layer features:
```bash
python extract_features.py \
  features/rwanda_vgg16_block5_pool \
  conv7 \
  --flatten \
  --model models/rwanda_vgg16_tuned_flattened.h5 \
  --input-type=features \
  --batch-size=16 \
  --output-dir=features/rwanda_vgg16_tuned_conv7/
```

### And then retrain the index predictors!

```bash
python train_index.py \
  features/rwanda_vgg16_tuned_conv7_flattened/ \
  csv/rwanda_DHS_wealth.csv \
  csv/rwanda_cluster_avg_educ_nightlights.csv \
  csv/rwanda_TL.csv \
  nightlights/F182010.v4d_web.stable_lights.avg_vis.tif \
  models/indexes/rwanda_vgg16_tuned \
  -v
```

## Getting images that cause activation in later layers

Extract the features of a convolutional layer:

```bash
python extract_features.py \
  features/rwanda_vgg16_block4_pool/ \
  block5_conv3 \
  --model models/rwanda_vgg16_tuned.h5
  --input-type=features
  --batch-size=16
  --output-dir=features/rwanda_vgg16_tuned_block5_conv3/
```

Compute which images activate each filter in that layer:

```bash
python get_activations.py \
  features/rwanda_vgg16_tuned_block5_conv3/ \
  activations/rwanda_vgg16_tuned_block5_conv3.txt \
  --exemplar-count=10
```

Visualize which images maximize each filter:

```bash
python visualize_activations.py \
  activations/rwanda_vgg16_tuned_block5_conv3.txt \
  images/Rwanda_simple/ \
  activations/rwanda_vgg16_tuned_block5_conv3.pdf
```

## Restricting feature extraction and training to a subset of images

Run this command twice.  Once to produce training indexes 
for training the top
(`indexes/Rwanda_train_top_training_indexes.txt`), and once
to produce indexes for fine-tuning
(`indexes/Rwanda_tuning_training_indexes.txt`).

```bash
python util/sample.py \
  images/Rwanda_simple/ \
  csv/rwanda_TL.csv \
  indexes/Rwanda_test_indexes.txt \
  10000 > indexes/Rwanda_train_top_training_indexes.txt
```

You can also precompute the indexes of features that are
close to the DHS clusters:

```bash
python util/geometry.py \
  csv/rwanda_DHS_wealth.csv \
  csv/rwanda_cluster_avg_educ_nightlights.csv \
  csv/rwanda_cluster_avg_water_nightlights.csv \
  csv/rwanda_TL.csv \
  nightlights/F182010.v4d_web.stable_lights.avg_vis.tif > indexes/Rwanda_dhs_cluster_indexes.txt
```

You can restric feature extraction to just examples with
indexes defined in a set of files:

```bash
python extract_features.py \
  images/Rwanda_simple/ \
  block5_pool \
  --output-dir features/rwanda_vgg16_block5_pool/ \
  --filter-indexes indexes/Rwanda_test_indexes.txt indexes/Rwanda_train_top_training_indexes.txt
```

Note that you can provide multiple files from which to read
the indexes for feature extraction!

Then, you can train the top or do fine-tuning with these
precomputed training indexes, for example:

```bash
python train_top.py \
  features/rwanda_vgg16_block5_pool \
  csv/rwanda_TL.csv \
  indexes/Rwanda_test_indexes.txt \
  --learning-rate 0.01 \
  --batch-size=100 \
  --epochs=6 \
  --training-indexes-file=indexes/Rwanda_train_top_training_indexes.txt \
  -v
```

## Setup

If you're setting up a machine from scratch, you'll need to follow these steps.

### GPU setup

It's critical that you work with a GPU for this to have any
reasonable performance on training and feature extraction.

On a Google Cloud compute instance:

Install the GPU (instructions based on those from (Google Cloud docs)[https://cloud.google.com/compute/docs/gpus/add-gpus]):

```bash
echo "Checking for CUDA and installing."
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda -y
fi
```

Add the CudNN library:
From your main development computer (not Google Cloud), transfer the `deb` that you download from [here](https://developer.nvidia.com/cudnn) (make sure to download version 5.1).
```bash
scp -i ~/.ssh/google_compute_engine ~/Downloads/libcudnn6_6.0.20-1+cuda8.0_amd64.deb  andrew@35.185.45.28:/home/andrew/
```

Then from the compute machine, install it with `dpkg`:
```bash
sudo dkpg -i libcudnn6_6.0.20-1+cuda8.0_amd64.deb
```

### Other Project dependencies

```bash
git clone <link to this repository>

sudo apt-get update
sudo apt-get install python-pip --fix-missing
pip install virtualenv
sudo apt-get install python-gdal

cd DiiD-Predictor
virtualenv --system-site-packages venv -p python3  # lets you access system-wide python3-gdal
source venv/bin/activate
pip install -I -r requirements.txt
```

## Get images for a country

```bash
mkdir images
cd images/
gsutil -m cp -r gs://diid/Rwanda .
```
