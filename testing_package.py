import pandas as pd
import mitigate_disparity

data = pd.read_csv('MLPE_example_dataset.csv')
mitigate_disparity.fit(data)
