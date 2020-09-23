import sklearn

# local imports
from sklearn.impute import SimpleImputer

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


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
