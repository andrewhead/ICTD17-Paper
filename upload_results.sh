#! /bin/bash
country=$1
SAVE_ROOT="gs://diid/backups"

if [ "$country" == "rwanda" ]
then
  lower_case_country="rwanda"
  UPPER_CASE_COUNTRY="Rwanda"
  BEST_TOP_MODEL=models/trained-top.20170902-232358.lr-0.0005.h5
  BEST_TUNED_MODEL=models/fine-tuned.20170912-171757.lr-1e-10.h5
  BEST_WHOLE_MODEL=models/long-tuned.20170911-231051.lr-1e-10.h5
elif [ "$country" == "haiti" ]
then
  lower_case_country=haiti
  UPPER_CASE_COUNTRY=Haiti
  BEST_TOP_MODEL=models/trained-top.20170831-191829.lr-6.25e-05.h5
  BEST_TUNED_MODEL=models/fine-tuned.20170915-185311.lr-1e-08.h5
  BEST_WHOLE_MODEL=models/long-tuned.20170910-022814.lr-1e-10.h5
elif [ "$country" == "nepal" ]
then
  UPPER_CASE_COUNTRY=Nepal
  lower_case_country=nepal
  BEST_TOP_MODEL=models/trained-top.20170904-062425.lr-0.002.h5
  BEST_TUNED_MODEL=models/fine-tuned.20170910-161917.lr-1e-08.h5
elif [ "$country" == "nigeria" ]
then
  UPPER_CASE_COUNTRY=Nigeria
  lower_case_country=nigeria
  BEST_TOP_MODEL=models/trained-top.20170903-075115.lr-0.002.h5
  BEST_TUNED_MODEL=models/fine-tuned.20170910-160027.lr-1e-07.h5
else
  echo "Did not recognize country"
  exit
fi


function upload {
  ITEM_NAME=$1
  REMOTE_PATH=$2
  LOCAL_PATH=$3
  echo "###########################################"
  echo "# Starting to upload $ITEM_NAME"
  echo "###########################################"
  echo ""
  if [ -z "$LOCAL_PATH" ]
  then
    echo "Path for $ITEM_NAME not specified.  Skipping."
  elif [[ $LOCAL_PATH != *"*"* ]] && [ ! -e "$LOCAL_PATH" ]
  then
     echo "Couldn't find file $LOCAL_PATH for item $ITEM_NAME"
  else
    echo "Uploading $ITEM_NAME from $LOCAL_PATH to $REMOTE_PATH"
    gsutil -q -m cp -r $LOCAL_PATH $SAVE_ROOT/$REMOTE_PATH
  fi
  echo ""
}


# Upload the trained models
upload "top trained model" best_top_models/$country/ $BEST_TOP_MODEL
upload "fine-tuned model" best_tuned_models/$country/ $BEST_TUNED_MODEL
upload "whole trained model" best_whole_models/$country/ $BEST_WHOLE_MODEL

# Upload the final-layer features extracted for each model
upload "fine-tuned features" \
  features/fine_tuned_conv7/$country/ \
  features/${lower_case_country}_revamp_vgg16_conv7_flattened.${BEST_TUNED_MODEL}
upload "whole model features" \
  features/whole_trained_conv7/$country/ \
  features/${lower_case_country}_whole_vgg16_conv7_flattened

# Upload the training logs
upload "training log" training_logs/$country/ nohup.out
upload "bash history" bash_histories/$country/ ~/.bash_history
upload "miscellaneous log files" training_logs/misc/$country/ "*.log"
upload "miscellaneous txt files" training_logs/misc/$country/ "*.txt"

# Upload the training and testing indexes
upload "DHS cluster image indexes" indexes/dhs_clusters/$country/ \
  indexes/${UPPER_CASE_COUNTRY}_dhs_cluster_indexes.txt
upload "training image indexes" indexes/training/$country/ \
  indexes/${UPPER_CASE_COUNTRY}_training.txt
upload "validation image indexes" indexes/validation/$country/ \
  indexes/${UPPER_CASE_COUNTRY}_validation.txt

# Upload the DHS data files
upload "DHS wealth data" dhs/$country/ csv/${country}_DHS_wealth.csv
upload "DHS education data" dhs/$country/ csv/${country}_cluster_avg_educ_nightlights.csv
upload "DHS water data" dhs/$country/ csv/${country}_cluster_avg_water_nightlights.csv
upload "DHS height data" dhs/$country/ csv/${country}_height_4_age.csv
upload "DHS weight data" dhs/$country/ csv/${country}_weight_4_age.csv
upload "DHS weight 4 height data" dhs/$country/ csv/${country}_weight_4_height.csv
upload "DHS female BMI data" dhs/$country/ csv/${country}_female_bmi.csv
upload "DHS bed-net data" dhs/$country/ csv/${country}_bed_net_num.csv
upload "DHS hemoglobin data" dhs/$country/ csv/${country}_hemoglobin.csv
upload "DHS electricity data" dhs/$country/ csv/${country}_electricity.csv
upload "DHS mobile data" dhs/$country/ csv/${country}_mobile.csv

# Upload the nightlights data
upload "Nighlights data" nightlights/ nightlights/F182010.v4d_web.stable_lights.avg_vis.tif 
