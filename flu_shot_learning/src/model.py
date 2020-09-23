# third party imports
import pandas
import tensorflow as tf
import numpy

from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from flu_shot_learning.src.data import TFDataTransformer
from flu_shot_learning.src.preprocessor import FeatureTypeImputer
from flu_shot_learning.src.feature_mapper import NormalizationInputEncoder, CategoricalInputEncoder, StringInputEncoder


def encode_input_features(data_set=None):
    """
    Apply preprocessing and feature mapping/encoding
    """
    all_inputs = []
    encoded_features = []
    # apply categorical encoder
    string_encoder = StringInputEncoder(CATEGORICAL_FEATURES)
    cat_inputs = string_encoder.create_inputs()
    cat_encoded_features = string_encoder.encode_input_features(cat_inputs, data_set)
    # apply numerical encoder
    numerical_encoder = NormalizationInputEncoder(NUMERICAL_FEATURES)
    num_inputs = numerical_encoder.create_inputs()
    num_encoded_features = numerical_encoder.encode_input_features(num_inputs, data_set)
    # prepare input, encoded features
    all_inputs.extend(cat_inputs)
    all_inputs.extend(num_inputs)
    encoded_features.extend(cat_encoded_features)
    encoded_features.extend(num_encoded_features)
    return all_inputs, encoded_features


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


class FluShotClassifier:
    """
    Class for Keras based FluShot classifier 
    """
    _BATCH_SIZE = 32
    _EPOCHS = 50
    _METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc')]

    def __init__(self):
        pass

    def setup_network(self, all_inputs=None, encoded_features=None):
        all_encoded_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(256, activation="relu")(all_encoded_features)
        x = tf.keras.layers.Dropout(0.25)(x)
        output = tf.keras.layers.Dense(2, activation="sigmoid")(x)
        model = tf.keras.Model(all_inputs, output)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=self._METRICS
        )
        return model

    def fit(self, xtrain, ytrain, xval, yval):
        train_data_set = TFDataTransformer().transform(xtrain, ytrain).batch(self._BATCH_SIZE)
        val_data_set = TFDataTransformer().transform(xval, yval).batch(self._BATCH_SIZE)
        all_inputs, encoded_features = encode_input_features(train_data_set)
        self.model = self.setup_network(all_inputs, encoded_features)
        self.model.fit(train_data_set, epochs=self._EPOCHS, validation_data=val_data_set)
        return self

    def predict(self, xtest=None):
        test_data_set =  tf.data.Dataset.from_tensor_slices(dict(xtest)).batch(self._BATCH_SIZE)
        return self.model.predict(test_data_set)


class FluShotModel:

    _BATCH_SIZE = 32
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


if __name__ == '__main__':
    TRAIN_DATA_FEATURES = pandas.read_csv("flu_shot_learning/data/training_set_features.csv").drop('respondent_id', axis=1)[ALL_FEATURES]
    TRAIN_DATA_LABELS = pandas.read_csv("flu_shot_learning/data/training_set_labels.csv").drop('respondent_id', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(TRAIN_DATA_FEATURES, TRAIN_DATA_LABELS, random_state=42)

    MODEL = FluShotModel()
    MODEL.fit(X_train, y_train, X_test, y_test)
