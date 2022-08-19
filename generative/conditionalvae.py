from tensorflow import keras

class VAE(keras.Model):
    def __init__(self, encoder, decoder, recons_loss_factor=1_000, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.recons_loss_factor = recons_loss_factor
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        images, labels = inputs[0]
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder((z, labels))
        return reconstruction, z_mean, z_log_var

    def compute_losses(self, data):
        image, digit = data[0]
        reconstruction, z_mean, z_log_var = self.call(data)

        # Reconstruction loss thorugh mse
        reconstruction_loss = tf.reduce_mean(tf.square(image - reconstruction), axis=[1, 2, 3])

        # Reconstruction loss through binary crossentropy
        # reconstruction_loss = keras.losses.binary_crossentropy(K.batch_flatten(data) , K.batch_flatten(reconstruction), axis=-1)

        # KL Divergence loss
        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - K.exp(z_log_var), axis=-1)

        # Total loss is sum of both losses
        total_loss = self.recons_loss_factor * reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss

    def train_step(self, data):
        """Defines how the model must be trained"""

        with tf.GradientTape() as tape:
            # For all operationes here, the graphs of the gradients will be recorded and stored in tape

            reconstruction_loss, kl_loss, total_loss = self.compute_losses(data)

            total_loss = K.mean(total_loss, axis=0)
            kl_loss = K.mean(kl_loss, axis=0)
            reconstruction_loss = K.mean(reconstruction_loss, axis=0)

        # Computes the gradient
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Changes the weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Updates the metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        reconstruction_loss, kl_loss, total_loss = self.compute_losses(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

