# DiiD-Predictor

## When deploying

* Install the virtualenv requirements
* Upload the existing ImageNet weights

## Setup

On a Google Cloud compute instance:

```bash
sudo apt-get install git
git clone <link to this repository>

sudo easy_install pip
sudo pip install virtualenv

virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Get images for a country

```bash
cd images/
gsutil -m cp -r gs://diid/Rwanda .
```

## Preprocess images to have the expected indexes

*Note*: The `rwanda_TL.csv` file comes will have to be uploaded securely (it's protected data).
For uploading this file with `scp`, see related instructions for connecting over `ssh` (here)[https://cloud.google.com/compute/docs/instances/connecting-to-instance#standardssh].
Then, run the `copy_images.py` script.

```bash
python copy_images.py csv/rwanda_TL.csv --output-dir images/Rwanda_simple/
```

## Split into the test set

```bash
python util/get_test_indexes.py images/Rwanda_simple/ 0.1 > indexes/Rwanda_test_indexes.txt
```

## Train!

Note that this steps depends on having run the previous steps to have the CSV file for Rwanda data, having the images moved into simple indexing form (single integer in the title), and having generated the test set image indexes.

```bash
python train.py csv/rwanda_TL.csv images/Rwanda_simple/ indexes/Rwanda_test_indexes.txt  -v
```
