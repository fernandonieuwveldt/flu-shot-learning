# third party imports
import pandas
import tensorflow as tf

from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from flu_shot_learning.src.preprocessor import FeatureTypeImputer, FluShotTransformer
from flu_shot_learning.src.classifier import FluShotClassifier


class FluShotModel:
    """
    Model class putting everything together
    """

    def __init__(self):
        pass

    def fit(self, xtrain, ytrain, xval, yval):
        """
        """
        self.transformer = FluShotTransformer()
        self.transformer.fit(xtrain)
        xtrain_data_frame = self.transformer.transform(xtrain)
        xval_data_frame = self.transformer.transform(xval)
        self.classifier = FluShotClassifier()
        self.classifier.fit(xtrain_data_frame, ytrain, xval_data_frame, yval)

    def predict(self, xtest=None):
        xtest_data_frame = self.transformer.transform(xtest)
        predictions = self.classifier.predict(xtest_data_frame)
        return predictions
