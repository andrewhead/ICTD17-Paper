# To run: awk -f [name_of_this_file] nohup.out
# The output can be pasted as values into a Google spreadsheet
BEGIN { 
  bs=""; lr=""; metric="";
  print "Metric\tTest index\tFold index\tScore"
}
/Test best score/ {
  match($0, /Test best score: (.*)/, score_res);
  score = score_res[1];
  print metric "\t" test_index "\t" fold_index "\t" score
  fold_index = fold_index + 1;
}
/Now predicting/ {
  old_metric = metric;
  match($0, /Now predicting (.* )\.\.\./, metric_res);
  metric = metric_res[1];
  if (metric != old_metric) {
    test_index = -1;
  }
  test_index = test_index + 1;
  fold_index = 0;
}
