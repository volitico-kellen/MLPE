import pandas as pd
import itertools

class MeasureDisparityMLPE():
    def __init__(self, patient_info:pd.core.frame.DataFrame):
        self.patient_info = patient_info.copy()

    def prepare_data(self, low_ci_scores):
        if len(self.patient_info) != len(low_ci_scores):
            raise Exception("patient_info and low_ci_scores must have the same length")

        # removing any non-string/object columns from dataframe
        demographic_attributes = self.patient_info.dtypes[self.patient_info.dtypes == 'O']
        demographic_attributes = list(set(demographic_attributes.index))
        self.patient_info = self.patient_info[demographic_attributes]
        self.patient_info['model_prediction'] = low_ci_scores

    def measure_disparity(self, level=1):
        groups = list(itertools.combinations(self.patient_info.columns[:-1], level))
        all_groupings = []
        for atts in groups:
            all_groupings.append(pd.DataFrame(self.patient_info.groupby(list(atts))['model_prediction'].mean()))
        disparity_df = pd.concat(all_groupings).sort_values(by='model_prediction').reset_index()
        disparity_df.columns = [f'demographic_{i + 1}' for i in range(level)] + ['model_prediction']
        return disparity_df


