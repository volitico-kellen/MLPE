# MLPE
Metric Lattice for Performance Estimation

Penultimate column is interpereted as labels and last column is interpereted as predictions.
Without any specification, MLPE fit() function will catagorize data with a prediction as test data and training data otherise;
and any numeric fields will be considered feature data and demographic attributes otherwise.



Ensure there are no NaN values in feature data (replace with median, mean, etc.). NaN is fine in demographic categories.
