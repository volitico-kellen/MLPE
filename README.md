# MLPE
Metric Lattice for Performance Estimation

Load in a single dataframe with optional definitions of demographic attributes and feature data

  -Predictions are last column of dataframe
  
  -Labels are penultimate column of dataframe
  
  Unless otherwise noted:
  
      -Demographic attributes are pandas type Object (string)
      -Feature data is everything else
      
      -Ensure there are no NaN values in feature data (replace with median, mean, etc.)
      
      -NaN is fine in demographic categories.
      
  Training data has NaN predictions
  
  Test data has non NaN predictions

Ensure there are no NaN values in feature data (replace with median, mean, etc.). NaN is fine in demographic categories.

prepare_example_dataset.py pulls synthetic data from the Synthea COVID-19 dataset and will produce example data that fits these criteria.
