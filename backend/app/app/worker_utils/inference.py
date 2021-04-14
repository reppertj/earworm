import tensorflow as tf
from app.core.config import settings


class Infer:
    def __init__(self):

        self.spec = tf.saved_model.load(settings.SPEC_MODEL_PATH).signatures[
            "serving_default"
        ]
        self.encode = tf.saved_model.load(settings.ENCODE_MODEL_PATH).signatures[
            "serving_default"
        ]
        self.embed = tf.saved_model.load(settings.EMBED_MODEL_PATH).signatures[
            "serving_default"
        ]

    def __call__(self, array_in):
        """Input should be numpy array of [1, 66150]"""
        sgram = self.spec(tf.convert_to_tensor(array_in))["output_0"]
        encoded = self.encode(sgram)
        # constant 4 here is particular to the embedding model
        embedded = self.embed(
            **{"0": encoded["output_0"], "1": tf.constant(4, dtype=tf.int64)}
        )["output_0"]
        return embedded.numpy()