"""
workflow for training and predicting
"""

import pandas
from sklearn.model_selection import train_test_split

from flu_shot_learning.src.model import FluShotModel
from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES


TRAIN_DATA_FEATURES = pandas.read_csv("flu_shot_learning/data/training_set_features.csv").drop('respondent_id', axis=1)[ALL_FEATURES]
TRAIN_DATA_LABELS = pandas.read_csv("flu_shot_learning/data/training_set_labels.csv").drop('respondent_id', axis=1)
X_train, X_test, y_train, y_test = train_test_split(TRAIN_DATA_FEATURES, TRAIN_DATA_LABELS, random_state=42)

MODEL = FluShotModel()
MODEL.fit(X_train, y_train, X_test, y_test)

def submit_run(file_name=None):
    submit_data_frame = pandas.DataFrame([], columns=['respondent_id', 'h1n1_vaccine','seasonal_vaccine'])
    submit_data_frame['respondent_id'] = RAW_DATA_TEST.index.values
    submit_data_frame[['h1n1_vaccine','seasonal_vaccine']] = predictions
    submit_data_frame.to_csv('flu_shot_learning/data/f"{file_name}".csv', index=False)
