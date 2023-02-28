# MLPE
Metric Lattice for Performance Estimation

Penultimate column is interpereted as labels and last column is interpereted as predictions.
Without any specification, MLPE fit() function will catagorize data with a prediction as test data and training data otherise;
and any columns as pandas objects 'O' (strings) will be considered demographic attributes and feature data otherwise.



Ensure there are no NaN values in feature data (replace with median, mean, etc.). NaN is fine in demographic categories.
