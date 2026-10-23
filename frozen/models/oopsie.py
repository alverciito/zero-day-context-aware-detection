import numpy as np
from .networks import (ThreeLayerMLP, LSTMAutoencoder, MLPAutoencoder, Conv1DAutoencoder, Conv2DAutoencoder,
                       MHAAutoencoder, StatisticalAnomaly)
NETWORK_TYPES = {
    'mha': MHAAutoencoder,
    'original': ThreeLayerMLP,
    'lstm': LSTMAutoencoder,
    'mlp': MLPAutoencoder,
    'conv1d': Conv1DAutoencoder,
    'conv2d': Conv2DAutoencoder,
    'stat': StatisticalAnomaly
}

def __param_check__(n_features, autoencoder_size, hidden_ratio, ae_type, output_ae_type,
                    sequence_length, learning_rate, fm_grace_period, ad_grace_period, execution_window, feature_map):
    """
    Check the parameters of the KitNET instance.
    :return: fm_grace_period.
    """
    if not isinstance(n_features, int):
        raise TypeError(f"n_features must be an integer, not {type(n_features)}")
    if n_features <= 0:
        raise ValueError(f"n_features must be a positive integer, not {n_features}")

    if not isinstance(autoencoder_size, int):
        raise TypeError(f"autoencoder_size must be an integer, not {type(autoencoder_size)}")

    if autoencoder_size <= 0:
        raise ValueError(f"autoencoder_size must be a positive integer, not {autoencoder_size}")

    if not isinstance(hidden_ratio, float):
        raise TypeError(f"hidden_ratio must be a float, not {type(hidden_ratio)}")

    if hidden_ratio <= 0.0 or hidden_ratio > 1.0:
        raise ValueError(f"hidden_ratio must be in the range (0.0, 1.0], not {hidden_ratio}")

    if not isinstance(ae_type, str):
        raise TypeError(f"ae_type must be a string, not {type(ae_type)}")

    if ae_type not in NETWORK_TYPES:
        raise ValueError(f"ae_type must be 'transformer' or 'original', not {ae_type}")

    if not isinstance(output_ae_type, str) and output_ae_type is not None:
        raise TypeError(f"output_ae_type must be a string, not {type(output_ae_type)}")

    if output_ae_type not in NETWORK_TYPES and output_ae_type is not None:
        raise ValueError(f"output_ae_type must be 'transformer' or 'original', not {output_ae_type}")

    if not isinstance(sequence_length, int):
        raise TypeError(f"sequence_length must be an integer, not {type(sequence_length)}")

    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, not {sequence_length}")

    if not isinstance(learning_rate, float):
        raise TypeError(f"learning_rate must be a float, not {type(learning_rate)}")

    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be a positive float, not {learning_rate}")

    if feature_map is not None:
        if not isinstance(feature_map, list):
            raise TypeError(f"feature_map must be a list, not {type(feature_map)}")

        for idx, fmap in enumerate(feature_map):
            if not isinstance(fmap, list) and not isinstance(fmap, np.ndarray):
                raise TypeError(f"feature_map[{idx}] must be a list, not {type(fmap)}")

            for feature in fmap:
                if not isinstance(feature, int) and not isinstance(feature, np.int_):
                    raise TypeError(f"feature_map[{idx}] must contain integers, not {type(feature)}")

                if feature < 0 or feature >= n_features:
                    raise ValueError(f"feature_map[{idx}] must contain integers in the range "
                                     f"[0, {n_features}), not {feature}")

    if not isinstance(ad_grace_period, int):
        raise TypeError(f"ad_grace_period must be an integer, not {type(ad_grace_period)}")

    if ad_grace_period <= 0:
        raise ValueError(f"ad_grace_period must be a positive integer, not {ad_grace_period}")

    if fm_grace_period is not None:
        if not isinstance(fm_grace_period, int):
            raise TypeError(f"fm_grace_period must be an integer, not {type(fm_grace_period)}")

        if fm_grace_period <= 0:
            raise ValueError(f"fm_grace_period must be a positive integer, not {fm_grace_period}")
    else:
        fm_grace_period = ad_grace_period

    if not isinstance(execution_window, int):
        raise TypeError(f"execution_window must be an integer, not {type(execution_window)}")

    if execution_window <= 0:
        raise ValueError(f"execution_window must be a positive integer, not {execution_window}")

    if feature_map is not None:
        fm_grace_period = 0

    return fm_grace_period


def __check_input__(x, n_features):
    """
    Check the input data.
    """
    if isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy array or a list, not {type(x)}")

    if x.shape[-1] != n_features:
        raise ValueError(f"x must have {n_features} features, not {x.shape[-1]}")

    return x
