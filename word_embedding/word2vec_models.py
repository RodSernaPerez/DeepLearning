from abc import ABC
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf

from tensorflow import keras


class Word2VecModel(keras.models.Model, ABC):
    def __init__(self, embedding_size: int, number_words: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.number_words = number_words

        self.layer_embedding = keras.layers.Embedding(self.number_words + 1, self.embedding_size, name="embedder")

        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def loss_function(self):
        raise NotImplementedError

    @tf.function
    def train_step(self, inputs: Dict):
        """Defines how the model must be trained
        :param inputs: dictionary with `words` and `contexts` as keywords
        :return: dict with losses and metrics scores
        """
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        with tf.GradientTape() as tape:
            # For all operations here, the graphs of the gradients will be recorded and stored in tape
            results = self.call(inputs)
            loss = self.loss_function(inputs, results)

        # Computes the gradient
        grads = tape.gradient(loss, self.trainable_weights)

        # Changes the weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Updates the metrics
        self.total_loss_tracker.update_state(loss)

        return {"loss": self.total_loss_tracker.result()}

    def get_embeddings(self, indices_words: np.ndarray):
        return self.layer_embedding(indices_words)

    def get_closest_words(self, index_word, n=10):
        """Given the index of a word, return the indices of the most similars."""
        e = self.get_embeddings(np.expand_dims(index_word, axis=0))
        similarities = [1 - scipy.spatial.distance.cosine(e, w) for w in self.layer_embedding.get_weights()[0]]
        ordered_words = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ordered_words[:n]


class CBOWModel(Word2VecModel):
    def __init__(self, embedding_size: int, number_words: int, **kwargs):
        super().__init__(embedding_size, number_words, **kwargs)

        # Layers
        self.mean_layer = keras.layers.GlobalAveragePooling1D(name="mean")
        self.decoder = keras.layers.Dense(self.number_words, activation="sigmoid", name="decoder")

    def call(self, inputs: Dict):
        e = self.layer_embedding(inputs["contexts"])
        m = self.mean_layer(e)
        y = self.decoder(m)
        return y

    @property
    def loss_function(self):
        def loss(inputs, predictions):
            x = keras.metrics.sparse_categorical_crossentropy(inputs["words"], predictions)
            return x

        return loss


class SkipgramModel(Word2VecModel):
    def __init__(self, embedding_size: int, number_words: int, **kwargs):
        super().__init__(embedding_size, number_words, **kwargs)

        # Layers
        self.flatten = keras.layers.Reshape((-1, ), name="flatten")
        self.decoder = keras.layers.Dense(self.number_words, activation="sigmoid", name="decoder")

    def call(self, inputs: Dict):
        e = self.layer_embedding(inputs["words"])
        e = self.flatten(e)  # Used to squeeze
        y = self.decoder(e)
        return y

    @property
    def loss_function(self):
        def sparse_multilabel_crossentropy(inputs, predicted_contexts):
            """
            Loss function for skipgram
            """
            contexts = inputs["contexts"]

            ones = tf.gather(params=predicted_contexts, indices=tf.cast(contexts, tf.int32), batch_dims=1)

            # Gathers the values of the words not in context
            a = 1 - tf.math.reduce_sum(tf.one_hot(indices=contexts, depth=self.number_words, dtype="int32"), axis=1)

            zeros = tf.boolean_mask(tensor=predicted_contexts, mask=a)

            loss = tf.reduce_sum(tf.math.log(ones)) + tf.reduce_sum(tf.math.log(1 - zeros)) / tf.cast(
                tf.shape(contexts)[0], dtype=tf.float32)
            return -loss / self.number_words

        return sparse_multilabel_crossentropy


class NegativeSamplingModel(Word2VecModel):
    def __init__(self, embedding_size: int, number_words: int, **kwargs):
        super().__init__(embedding_size, number_words, **kwargs)

        # Layers
        self.mean = keras.layers.GlobalAveragePooling1D(name="mean")
        self.flatten = keras.layers.Flatten(name="flatten")
        self.cosine = keras.layers.Dot(axes=-1, normalize=True)
        self.final_layer = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: Dict):
        e_words = self.layer_embedding(inputs["words"])
        e_words = self.flatten(e_words)  # Used to squeeze

        e_contexts = self.layer_embedding(inputs["contexts"])
        e_contexts = self.mean(e_contexts)

        cosine = self.cosine([e_contexts, e_words])
        y = self.final_layer(cosine)

        return y

    @property
    def loss_function(self):
        def loss(inputs, predictions):
            x = keras.metrics.binary_crossentropy(inputs["labels"], predictions[:, 0])
            return x

        return loss
