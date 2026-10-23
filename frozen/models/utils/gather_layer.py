import tensorflow as tf


class GatherLayer(tf.keras.layers.Layer):
    def __init__(self, indices, axis=-1, **kwargs):
        """
        Layer that gathers the inputs.
        :param indices: The indices to gather.
        :param axis: The axis to gather.
        :param kwargs: The keyword arguments.
        """
        super().__init__(**kwargs)
        self.indices = indices
        self.axis = axis

    def call(self, inputs, **kwargs):
        """
        Call method for the layer.
        :param inputs: The inputs.
        :param kwargs: The keyword arguments.
        :return: The gathered inputs.
        """
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

    def get_config(self):
        """
        Soporte para la serialización de la capa.
        """
        config = super(GatherLayer, self).get_config()
        config.update({
            'indices': self.indices,
            'axis': self.axis
        })
        return config

    def __call__(self, *args, **kwargs):
        """
        Call method for the layer.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The gathered inputs.
        """
        return super(GatherLayer, self).__call__(*args, **kwargs)
