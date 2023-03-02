import pandas as pd, numpy as np
import requests
import zipfile
import io
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# retrieving and unzipping data
r = requests.get('https://synthetichealth.github.io/synthea-sample-data/downloads/10k_synthea_covid19_csv.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))

# reading in data sources
conditions = pd.read_csv(z.open('10k_synthea_covid19_csv/conditions.csv'))
patients = pd.read_csv(z.open('10k_synthea_covid19_csv/patients.csv'))
observations = pd.read_csv(z.open('10k_synthea_covid19_csv/observations.csv'))

def age_bracket(date):
    timestamp = datetime.strptime(date,'%Y-%m-%d').timestamp()
    # 31536000 seconds in a year
    age = (time.time()-timestamp)/31536000
    if age < 18:
        return 'Child'
    elif age < 65:
        return 'Adult'
    else:
        return 'Senior'

# creating patient dataset for demographic attributes
pdf = patients[['Id','BIRTHDATE','COUNTY','RACE','ETHNICITY','GENDER']].copy()
pdf.loc[:, 'AGE_BRACKET'] = pdf.apply(lambda t: age_bracket(t['BIRTHDATE']), axis=1)
pdf.drop(['BIRTHDATE'], axis=1, inplace=True)

# creating observations dataset for data (features)
observations[observations['TYPE'] == 'numeric'].head()
odf_unpivoted = observations[observations['TYPE']=='numeric'].reset_index(drop=True)
odf_unpivoted.loc[:,'VALUE_NUMERIC'] = np.array(odf_unpivoted.loc[:, 'VALUE'].values, dtype=float)
odf = pd.pivot_table(odf_unpivoted,
                     values='VALUE_NUMERIC',
                     index=['PATIENT'],
                     columns=['DESCRIPTION'],
                     aggfunc='mean'
                     )

# removing features with over 90 percent NaNs
keep_column = (odf.isna().sum()/len(odf)) < 0.9
columns = list(keep_column[keep_column].index)
odf = odf[columns].copy()

odf.fillna(value=odf.mean(), inplace=True)
odf = odf.reset_index()
odf.describe().to_csv('data_summary.csv')
scaler = StandardScaler()
data_columns = odf.columns[1:]
odf = pd.concat([odf[['PATIENT']],pd.DataFrame(scaler.fit_transform(odf.iloc[:,1:]),columns=data_columns)], axis=1)

# creating labels
has_covid = set(conditions[conditions['DESCRIPTION'].isin(['COVID-19'])]['PATIENT'])
covid_df = pdf.merge(odf, left_on='Id', right_on='PATIENT', how='inner')
index_to_patient_mapping = covid_df[['Id']]
covid_df['has_covid'] = [i in has_covid for i in covid_df['Id']]
covid_df.drop(['Id', 'PATIENT'], axis=1, inplace=True)

# removing most informative data since the model is near perfect with all data

covid_df.drop(['Oxygen saturation in Arterial blood',
               'Body temperature',
               'Respiratory rate',
               'DALY', 'QALY', 'QOLS'], axis=1, inplace=True)

# training model to get predictions
X_model = covid_df.iloc[:,5:-1]
y_model = covid_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.25, random_state=0)
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs=-1)
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

covid_df.loc[:, 'prediction'] = np.nan
covid_df.loc[X_test.index, 'prediction'] = prediction


covid_df.to_csv('MLPE_example_dataset.csv', index=False)

