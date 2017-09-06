# To run: awk -f [name_of_this_file] nohup.out
# The output can be pasted as values into a Google spreadsheet
BEGIN { bs=""; lr=""; metric=""; }
/Average/ { 
  r2[metric] = $3;
  if ( metric ~ /water/ ) {
    print bs "\t" lr "\t" r2["wealth"] "\t" r2["education"] "\t" r2["water"]
  }
}
/batch size/ {
  match($0, /batch size ([0-9]+)/, bs_res);
  match($0, /learning rate ([^ ]*) /, lr_res);
  bs = bs_res[1];
  lr = lr_res[1];
}
/Now predicting/ {
  match($0, /Now predicting ([a-z]+)\.\.\./, metric_res);
  metric=metric_res[1];
}
