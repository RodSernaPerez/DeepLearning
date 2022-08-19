import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_nlp


class MLM(keras.models.Model):
    """
    Implementation of masked language model using BERT.
    Details can be seen in: https://arxiv.org/pdf/1810.04805.pdf
    """

    @property
    def params(self):
        return {"embedding_size": self.embedding_size,
                "number_heads": self.number_heads,
                "hidden_layers": self.hidden_layers,
                "mask_selection_proba": self.mask_selection_proba,
                "mask_token_rate": self.mask_token_rate,
                "random_token_rate": self.random_token_rate,
                "local_mask_width": self.local_mask_width,
                "vocabulary": self.tokenizer.get_vocabulary()
                }

    def __init__(self, embedding_size: int, number_heads: int, hidden_layers: int,
                 maximum_number_tokens: int = 512,
                 mask_selection_proba=0.15, mask_token_rate=0.8, random_token_rate=0.1,
                 tokenizer: keras.layers.TextVectorization = None, texts_fitting_tokenizer=None,
                 number_words: int = None, reuse_input_embeddings=True, local_mask_width=None,
                 **kwargs):
        """
        Builds the model
        :param embedding_size: size of the word embeddings
        :param number_heads: number of heads in the transformers layers. Must be a divisor of embedding_size
        :param hidden_layers: number of layers of transformers
        :param mask_selection_proba: probability of a token being replaced
        :param mask_token_rate: probability of a selected token being replaced by [MASK].
        mask_token_rate + random_token_rate must be less than one. Default value obtained by the original paper.
        :param random_token_rate: probability of a selected token being replaced by another token.
        mask_token_rate + random_token_rate must be less than one. Default value obtained by the original paper.
        :param tokenizer: optional, a fitted TextVectorization layer
        :param texts_fitting_tokenizer: optional, not needed when a tokenizer is provided. Texts for
        fitting the tokenizer.
        :param number_words: optional, not needed when a tokenizer is provided. Maximum number of words to consider.
        :param reuse_input_embeddings optional, reuse input embeddings in the last layer.
        (see https://arxiv.org/pdf/1608.05859.pdf)
        :param local_mask_width: integer indicating the width of the mask window. If None, not local mask is used.
        :param kwargs: other arguments passed to keras.models.Model
        """
        super(MLM, self).__init__(**kwargs)
        self.number_words = number_words
        self.embedding_size = embedding_size
        self.number_heads = number_heads
        self.mask_selection_proba = mask_selection_proba
        self.hidden_layers = hidden_layers
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.reuse_input_embeddings = reuse_input_embeddings
        self.maximum_number_tokens = maximum_number_tokens
        self.local_mask_width = local_mask_width

        if tokenizer is None:
            if texts_fitting_tokenizer is None:
                raise ValueError("If a tokenizer is not provided, texts_fitting_tokenizer  is needed.")
            self.tokenizer = keras.layers.TextVectorization(max_tokens=number_words)
            self.tokenizer.adapt(texts_fitting_tokenizer)
        else:
            self.tokenizer = tokenizer

        self.truncation_layer = keras.layers.Lambda(function=lambda x: x[:, -self.maximum_number_tokens:],
                                                    name="trucation_layer")
        self.number_words = len(tokenizer.get_vocabulary())

        self.total_loss_tracker = keras.metrics.Mean(name="categorical_crossentropy")
        self.acc_metric_tracker = keras.metrics.Mean(name="categorical_accuracy")

        self.categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()
        self.categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Builds layers
        self.masker = keras_nlp.layers.MLMMaskGenerator(vocabulary_size=self.number_words + 1,  # Because of MASK token
                                                        mask_selection_rate=mask_selection_proba,
                                                        mask_token_id=self.number_words,
                                                        unselectable_token_ids=[0, 1],  # Padding and UNK
                                                        mask_token_rate=mask_token_rate,
                                                        random_token_rate=random_token_rate
                                                        )

        self.word_pos_embedding_layer = self._build_embedding_layer()

        # Transformers
        self.transformers = self._build_transformers_layers()

        # FFN layers
        self.ffn_layers = self._builds_ffn_layers()

    def _builds_ffn_layers(self):
        """Builds feed forward final layers."""
        mlm_head_dense_1 = keras.layers.Dense(self.embedding_size, activation="relu", name="mlm_head_dense_1")
        normalization_layer = keras.layers.LayerNormalization()
        mlm_head_dense_2 = keras.layers.Dense(self.number_words, activation="softmax", name="mlm_head_dense_2")
        if self.reuse_input_embeddings:
            mlm_head_dense_2.build(input_shape=(self.embedding_size,))
            mlm_head_dense_2.weights[0] = tf.transpose(
                self.word_pos_embedding_layer.get_layer("word_embedding").weights[0][:-1, :])
        return keras.models.Sequential(
            [mlm_head_dense_1,
             normalization_layer,
             mlm_head_dense_2], name="ffn_layers")

    def _build_embedding_layer(self):
        """Builds a word embedding layer with sinusoidal positional encoding."""

        # Embedding for words and positions
        input_for_embedding = keras.layers.Input(shape=(None,))
        word_embedder = keras.layers.Embedding(self.number_words + 1, self.embedding_size, name="word_embedding")
        pos_embedder = keras_nlp.layers.SinePositionEncoding(name="pos_embedding")
        we = word_embedder(input_for_embedding)
        position_embeddings = pos_embedder(we)

        we = we + position_embeddings
        return keras.models.Model(input_for_embedding, we, name="word_pos_embedding")

    def _build_transformers_layers(self):
        """Builds the hidden layers of transformers."""
        input_for_transformers = keras.layers.Input(shape=(None, self.embedding_size))
        input_for_padding = keras.layers.Input(shape=(None,), dtype=tf.bool)

        transformers = [keras_nlp.layers.TransformerEncoder(intermediate_dim=self.embedding_size,
                                                            num_heads=self.number_heads, name=f"transformer_{i + 1}")
                        for i in range(self.hidden_layers)]

        n_tokens = input_for_padding.shape[1]
        if self.local_mask_width and n_tokens:
            attention_mask = self._create_local_mask(n_tokens, self.local_mask_width)
        else:
            attention_mask = None

        x = input_for_transformers
        for h in transformers:
            y = h(x, padding_mask=input_for_padding, attention_mask=attention_mask)
            x = y + x

        return keras.models.Model([input_for_transformers, input_for_padding], x, name="transformers_layers")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.acc_metric_tracker
        ]

    def call(self, input_texts):
        """Produces word encodings on a batch of texts
        :param input_texts: numpy array of sentences in plain text
        :return: tensor of size [number of texts x None x embedding size]
        """
        input_ids = self.tokenizer(input_texts)
        word_embedding = self.word_pos_embedding_layer(input_ids)
        x = self.transformers([word_embedding, input_ids == 0])
        return x

    def _create_local_mask(self, shape, width):
        m = np.zeros(shape=(shape, shape))
        for i in range(shape):
            for j in range(shape):
                if abs(i - j) <= width:
                    m[i][j] = 1
        return tf.convert_to_tensor(m) == 1

    def compute_losses(self, input_texts):
        """Computes the list losses and metrics for some texts.
        :param input_texts: numpy array of sentences in plain text
        :return: dict with losses and metrics scores
        """

        input_ids = self.tokenizer(input_texts)
        input_ids = self.truncation_layer(input_ids)

        masked_tokens, masked_positions, masked_ids, _ = self.masker(input_ids).values()
        word_embedding = self.word_pos_embedding_layer(masked_tokens)

        encoded_words = self.transformers([word_embedding, input_ids == 0])

        masked_encoded_words = tf.gather(encoded_words, masked_positions, batch_dims=1)

        # Changes the batch size to be each masked token a row
        masked_encoded_words_flat = masked_encoded_words.merge_dims(0, 1)

        # Generates predictions
        predictions = self.ffn_layers(masked_encoded_words_flat)

        # Compute losses and metrics
        crossentropy_loss = self.categorical_crossentropy(masked_ids.merge_dims(0, 1), predictions)
        accuracy_metric = self.categorical_accuracy(masked_ids.merge_dims(0, 1), predictions)

        return {"loss": crossentropy_loss, "accuracy": accuracy_metric}

    def train_step(self, data):
        """Defines how the model must be trained
        :param data: numpy array of sentences in plain text
        :return: dict with losses and metrics scores
        """

        with tf.GradientTape() as tape:
            # For all operations here, the graphs of the gradients will be recorded and stored in tape
            r = self.compute_losses(data)

            crossentropy_loss = r["loss"]
            accuracy_metric = r["accuracy"]

        # Computes the gradient
        grads = tape.gradient(crossentropy_loss, self.trainable_weights)

        # Changes the weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Updates the metrics
        self.total_loss_tracker.update_state(crossentropy_loss)
        self.acc_metric_tracker.update_state(accuracy_metric)

        return {"loss": self.total_loss_tracker.result(),
                "accuracy": self.acc_metric_tracker.result()}

    def apply_mask(self, texts):
        """
        Applies tokenization and mask on a batch of texts
        :param texts: list of texts
        :return: tensor of ids of the tokens (with mask) and the ids of the masked tokens.
        """
        input_ids = self.tokenizer(texts)
        masked_tokens, _, masked_ids, _ = self.masker(input_ids).values()
        return masked_tokens, masked_ids

    def test_step(self, data):
        """
        Computes metrics on validation data
        :param data: numpy array of sentences in plain text
        :return: dict with losses and metrics scores
        """
        r = self.compute_losses(data)

        crossentropy_loss = r["loss"]
        accuracy_metric = r["accuracy"]

        self.total_loss_tracker.update_state(crossentropy_loss)
        self.acc_metric_tracker.update_state(accuracy_metric)

        return {"loss": self.total_loss_tracker.result(),
                "accuracy": self.acc_metric_tracker.result()}

    def predict(self, input_texts_with_mask):
        """
        Given a batch of masked sentences, return the probability of the masked words being each word in the vocab.
        :param input_texts_with_mask: array of texts with contain mask words.
        :return: tensor of size [number of texts x None x number words in vocabulary]
        """

        word_embedding = self.word_pos_embedding_layer(input_texts_with_mask)

        encoded_words = self.transformers([word_embedding, input_texts_with_mask == 0])

        # Finds the positions of the masked words
        masked_positions = tf.map_fn(tf.where, input_texts_with_mask == self.number_words,
                                     fn_output_signature=tf.RaggedTensorSpec(dtype=tf.int64, ragged_rank=0))
        masked_encoded_words = tf.gather(indices=masked_positions, params=encoded_words, batch_dims=1)

        # Changes the batch size to be each masked token a row
        masked_encoded_words_flat = masked_encoded_words.merge_dims(0, 1)

        predictions_flat = self.ffn_layers(masked_encoded_words_flat)

        # Get the original shape
        predictions = tf.RaggedTensor.from_row_lengths(values=predictions_flat,
                                                       row_lengths=masked_encoded_words.row_lengths())
        return predictions

    def get_baseline_scores(self, input_texts, batch_size=8):
        """
        Computes scores with a baseline model.
        The baseline predicts:
            - when the token has been masked, predicts the most frequent token
            - when the token has not been masked, predicts itself
        :param input_texts: array of texts to evaluate
        :param batch_size: batch size
        :return: dictionary with metrics
        """

        self.total_loss_tracker.reset_state()
        self.acc_metric_tracker.reset_state()

        for i in range(0, len(input_texts), batch_size):
            input_texts_batch = input_texts[batch_size * i: batch_size * (i + 1)]
            input_ids = self.tokenizer(input_texts)
            masked_tokens, masked_positions, masked_ids, _ = self.masker(input_ids).values()

            x = tf.gather(masked_tokens, masked_positions, batch_dims=1)
            x = x.merge_dims(0, 1)

            predictions_baseline = tf.where(x == self.number_words, x=2, y=x)
            predictions_baseline = tf.one_hot(predictions_baseline, depth=self.number_words)

            # Compute losses and metrics
            crossentropy_loss = self.categorical_crossentropy(masked_ids.merge_dims(0, 1), predictions_baseline)
            accuracy_metric = self.categorical_accuracy(masked_ids.merge_dims(0, 1), predictions_baseline)

            self.total_loss_tracker.update_state(crossentropy_loss)
            self.acc_metric_tracker.update_state(accuracy_metric)

        return {"loss": self.total_loss_tracker.result().numpy(),
                "accuracy": self.acc_metric_tracker.result().numpy()}
