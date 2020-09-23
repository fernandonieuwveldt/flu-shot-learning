import sklearn
import pandas
# local imports
from sklearn.impute import SimpleImputer

from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES


# This will replaced by a custom keras preprocessing layer
class FeatureTypeImputer(TransformerMixin, sklearn.base.BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.pipeline = ColumnTransformer([
                                ("numerical_imputer",  SimpleImputer(strategy='mean'), NUMERICAL_FEATURES),
                                ("categorical_imputer", SimpleImputer(strategy='constant', fill_value='missing'), CATEGORICAL_FEATURES)
                                ])
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class FluShotTransformer:
    def __init__(self):
        pass

    def __column_type_formatter(self, data_array=None):
        """
        Type mapper
        """
        data_frame = pandas.DataFrame(data_array, columns=ALL_FEATURES)
        data_frame[NUMERICAL_FEATURES] = data_frame[NUMERICAL_FEATURES].astype('float64')
        data_frame[CATEGORICAL_FEATURES] = data_frame[CATEGORICAL_FEATURES].astype('str')
        return data_frame

    def fit(self, xtrain, ytrain=None):
        self.imputer = FeatureTypeImputer()
        self.imputer.fit(xtrain)
        return self

    def transform(self, xtest=None):
        xtest_transformed_array = self.imputer.transform(xtest)
        return self.__column_type_formatter(xtest_transformed_array)
