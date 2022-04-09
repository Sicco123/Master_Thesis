import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
import data_simulation
import numpy as np
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Flatten  # to flatten the input shape

#
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
#
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon
#
#
# class Encoder(layers.Layer):
#     """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
#
#     def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
#         super(Encoder, self).__init__(name=name, **kwargs)
#         self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
#         self.dense_mean = layers.Dense(latent_dim)
#         self.dense_log_var = layers.Dense(latent_dim)
#         self.sampling = Sampling()
#
#     def call(self, inputs):
#         x = self.dense_proj(inputs)
#         z_mean = self.dense_mean(x)
#         z_log_var = self.dense_log_var(x)
#         z = self.sampling((z_mean, z_log_var))
#         return z_mean, z_log_var, z
#
#
# class Decoder(layers.Layer):
#     """Converts z, the encoded digit vector, back into a readable digit."""
#
#     def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
#         super(Decoder, self).__init__(name=name, **kwargs)
#         self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
#         self.dense_output = layers.Dense(original_dim, activation="sigmoid")
#
#     def call(self, inputs):
#         x = self.dense_proj(inputs)
#         return self.dense_output(x)
#
#
# class VariationalAutoEncoder(keras.Model):
#     """Combines the encoder and decoder into an end-to-end model for training."""
#
#     def __init__(
#             self,
#             original_dim,
#             intermediate_dim=64,
#             latent_dim=32,
#             name="autoencoder",
#             **kwargs
#     ):
#         super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
#         self.original_dim = original_dim
#         self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
#         self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)
#
#     def call(self, inputs):
#         z_mean, z_log_var, z = self.encoder(inputs)
#         reconstructed = self.decoder(z)
#         # Add KL divergence regularization loss.
#         kl_loss = -0.5 * tf.reduce_mean(
#             z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
#         )
#         self.add_loss(kl_loss)
#         return reconstructed


class last_layer(layers.Layer):
    def __init__(self, number_of_quantiles=1, activation=None, **kwargs):
        super(last_layer, self).__init__()
        self.activation = activations.get(activation)
        self.number_of_quantiles = number_of_quantiles

    def build(self, input_shape):
        self.input_shape_1 = input_shape[-1]
        ### Build delta
        delta_coef_matrix = tf.Variable(tf.random.normal(shape = [self.input_shape_1, self.number_of_quantiles]))
        delta_0_matrix = tf.Variable(tf.random.normal(shape=[1, self.number_of_quantiles]))
        delta_matrix = tf.concat([delta_0_matrix, delta_coef_matrix], axis=0)

        self.beta_matrix = tf.transpose(tf.cumsum(tf.transpose(delta_matrix)))
        delta_vec = delta_matrix[1:(self.input_shape_1+1), 1:self.number_of_quantiles]
        delta_0_vec = delta_matrix[0, 1:self.number_of_quantiles]
        delta_minus_vec = tf.maximum(0, -delta_vec)
        self.delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, 0)

        self.delta_0_vec_clipped = tf.clip_by_value(delta_0_vec,
                                                    clip_value_min=tf.reshape(self.delta_minus_vec_sum,
                                                                              delta_0_vec.shape),
                                                    clip_value_max=tf.convert_to_tensor(
                                                        (np.ones(np.shape(delta_0_vec)) * np.inf), dtype = 'float32'))

    def call(self, inputs, **kwargs):
        predicted_y_no_penalty = tf.matmul(inputs, self.beta_matrix[1:(self.input_shape_1 + 1)])
        predicted_y_modified = predicted_y_no_penalty + tf.cumsum(tf.concat([self.beta_matrix[1, 1],
                                                                             self.delta_0_vec_clipped], axis=1),
                                                                  axis=1)
        print(predicted_y_modified)
        return predicted_y_no_penalty, predicted_y_modified



class QRNN(tf.keras.Model):
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(units=hidden_dim_1, activation="sigmoid")  # Choose correct number of units
        self.layer_3 = Dense(units=hidden_dim_2, activation="sigmoid")
        self.layer_4 = last_layer(number_of_quantiles=output_dim, activation="sigmoid")
        # possible add "Dropout" to prevent overfitting

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        y_no_penalty, predicted_y_modified = self.layer_4(x)
        return y_no_penalty, predicted_y_modified


def objective_function(predicted_y, output_y, quantiles):
    quantile_length = len(quantiles)
    quantiles_tf = tf.convert_to_tensor(quantiles)

    output_y_tiled = tf.tile(output_y, (quantile_length, 1))
    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), output_y_tiled.shape)
    diff_y = output_y_tiled - predicted_y_tiled

    quantile_loss = tf.reduce_mean(diff_y * (quantiles_tf - (tf.sign(-diff_y) + 1) / 2))

    objective_function_value = quantile_loss

    return objective_function_value


def main():
    original_dim = 640
    learning_rate = 0.005
    quantiles = [0.1, 0.5, 0.9]
    nmqn = QRNN(4, 4, len(quantiles))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss_fn = lambda x, z: objective_function(x, z, quantiles)

    loss_metric = tf.keras.metrics.Mean()

    x_train, y_train = data_simulation.moon_data(original_dim, 1)
    data = np.column_stack((y_train, x_train))

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)


    epochs = 2

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            y_train_batch = tf.gather(batch_train, indices = 0, axis =1)
            x_train_batch = tf.gather(batch_train, indices = 1, axis =1)

            with tf.GradientTape() as tape:
                predicted_y, predicted_y_modified = nmqn(x_train_batch)
                # Compute reconstruction loss
                loss = loss_fn(predicted_y, y_train_batch)
                #loss += sum(nmqn.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, nmqn.trainable_weights)
            optimizer.apply_gradients(zip(grads, nmqn.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))


if __name__ == "__main__":
    main()
