# third party imports
import tensorflow as tf

from flu_shot_learning.src.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from flu_shot_learning.src.feature_mapper import NormalizationInputEncoder, StringInputEncoder


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
