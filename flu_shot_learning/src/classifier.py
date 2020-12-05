# third party imports
import tensorflow as tf

# local imports
from flu_shot_learning.src.data import TFDataMapper
from flu_shot_learning.src.transformer import FeatureTransformer


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
        train_data_set = TFDataMapper().transform(xtrain, ytrain).batch(self._BATCH_SIZE)
        val_data_set = TFDataMapper().transform(xval, yval).batch(self._BATCH_SIZE)
        all_inputs, encoded_features = FeatureTransformer().transform(train_data_set)
        self.model = self.setup_network(all_inputs, encoded_features)
        self.model.fit(train_data_set, epochs=self._EPOCHS, validation_data=val_data_set)
        return self

    def predict(self, xtest=None):
        test_data_set =  tf.data.Dataset.from_tensor_slices(dict(xtest)).batch(self._BATCH_SIZE)
        return self.model.predict(test_data_set)