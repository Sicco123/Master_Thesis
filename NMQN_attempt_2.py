import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations
import data_simulation
import numpy as np
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Flatten  # to flatten the input shape
import matplotlib.pyplot as plt


class last_layer(layers.Layer):
    def __init__(self, number_of_quantiles=1, activation=None, **kwargs):
        super(last_layer, self).__init__()
        self.activation = activations.get(activation)
        self.number_of_quantiles = number_of_quantiles

    def build(self, input_shape):

        self.input_shape_1 = input_shape[-1]
        ### Build delta
        self.delta_coef_matrix = tf.Variable(tf.random.normal(shape = [self.input_shape_1, self.number_of_quantiles]))
        delta_0_matrix = tf.Variable(tf.random.normal(shape=[1, self.number_of_quantiles]))
        delta_matrix = tf.concat([delta_0_matrix, self.delta_coef_matrix], axis=0)

        self.beta_matrix = tf.transpose(tf.cumsum(tf.transpose(delta_matrix)))
        delta_vec = delta_matrix[1:(self.input_shape_1+1), 1:self.number_of_quantiles]
        self.delta_0_vec = delta_matrix[0, 1:self.number_of_quantiles]
        delta_minus_vec = tf.maximum(0, -delta_vec)
        self.delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, 0)

        self.delta_0_vec_clipped = tf.clip_by_value(self.delta_0_vec,
                                                    clip_value_min=tf.reshape(self.delta_minus_vec_sum,
                                                                              self.delta_0_vec.shape),
                                                    clip_value_max=tf.convert_to_tensor(
                                                        (np.ones(np.shape(self.delta_0_vec)) * np.inf), dtype = 'float32'))

    def call(self, inputs, **kwargs):
        predicted_y_no_penalty = tf.matmul(inputs, self.beta_matrix[1:(self.input_shape_1 + 1),:]) + self.beta_matrix[1,:]

        predicted_y_modified = tf.matmul(inputs, self.beta_matrix[1:(self.input_shape_1 + 1),:]) + tf.cumsum(tf.concat([self.beta_matrix[1:2, 1],
                                                                             self.delta_0_vec_clipped], axis=0),
                                                                  axis=0)

        l1_penalty = tf.reduce_mean(self.delta_coef_matrix**2)
        self.add_loss(l1_penalty)

        delta_constraint = self.delta_0_vec_clipped - self.delta_minus_vec_sum
        delta_clipped = tf.clip_by_value(delta_constraint, clip_value_min=10**(-20), clip_value_max=np.Inf)

        delta_penalty = tf.reduce_mean(tf.abs(delta_clipped))#tf.reduce_mean(tf.abs(self.delta_0_vec - self.delta_0_vec_clipped))
        self.add_loss(delta_penalty)
        return predicted_y_no_penalty, predicted_y_modified



class QRNN(tf.keras.Model):
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(units=hidden_dim_1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L1(1))  # Choose correct number of units
        self.layer_3 = Dense(units=hidden_dim_2, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L1(1))
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
    quantile_tf = tf.convert_to_tensor(quantiles, dtype='float32')
    quantile_tf_tiled = tf.repeat(tf.transpose(quantile_tf), [len(output_y)])

    output_y = tf.cast(output_y, dtype = 'float32')
    output_y_tiled = tf.tile(output_y, [quantile_length])

    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), output_y_tiled.shape)
    diff_y = output_y_tiled - predicted_y_tiled

    quantile_loss = tf.reduce_mean(diff_y * (quantile_tf_tiled - (tf.sign(-diff_y) + 1) / 2))

    objective_function_value = quantile_loss

    return objective_function_value


def main():
    original_dim = 640
    learning_rate = 0.01
    penalty_1 = 0.0005
    penalty_2 = 5
    quantiles = [0.1, 0.5, 0.9]
    nmqn = QRNN(16, 16, len(quantiles))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss_fn = lambda x, z: objective_function(x, z, quantiles)

    loss_metric = tf.keras.metrics.Mean()

    x_train, true_quantiles = data_simulation.simulate_gaussian_data(original_dim, quantiles)#data_simulation.moon_data(original_dim, 1)
    y_train = x_train

    data = np.column_stack((y_train, x_train))

    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)


    epochs = 2000

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

                loss += penalty_1*sum(nmqn.losses[0:3])+penalty_2*(nmqn.losses[3])  # Add KLD regularization loss

            grads = tape.gradient(loss, nmqn.trainable_weights)
            optimizer.apply_gradients(zip(grads, nmqn.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    print("Evaluate")
    y, y_modified = nmqn(x_train)

    plt.plot(x_train)
    plt.plot(y_modified[:,0])
    plt.plot(true_quantiles[:,0])
    plt.show()

    print(y[:,0])
    print(y_modified[:,0])
    print(y_modified[:,1])
    print(y_modified[:,2])

if __name__ == "__main__":
    main()
