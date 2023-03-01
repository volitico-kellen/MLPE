import pandas as pd
import mitigate_disparity

data = pd.read_csv('MLPE_example_dataset.csv')
mlpe = mitigate_disparity.MLPE()
mlpe.fit(data)
mlpe.feedback()

