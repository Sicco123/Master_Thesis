import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras import activations
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense # for creating regular densely-connected NN layers.
from keras.layers import Flatten # to flatten the input shape
from keras import backend as K
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects
import activation_funcs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import data_simulation
import qrnn_plots
# kernel regularization L2 function

class simple_dense(Layer):
    def __init__(self, units =32, activation = None):
        super(simple_dense, self).__init__()
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        theta_init = tf.random_normal_initializer()
        self.theta = tf.Variable(initial_value = theta_init(shape = (input_shape[-1], self.units),
                                 dtype = 'float32') ,
                                 trainable = True)
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value = bias_init(shape=(self.units,),
                                dtype = 'float32'),
                                trainable = True)

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.theta)+self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class output_layer(Layer):
    def __init__(self, units = 32, activation = None, number_of_quantiles = 1):
        super(output_layer, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.number_of_quantiles = number_of_quantiles

    def build(self, input_shape):
        ### Build Theta weights
        theta_init = tf.random_normal_initializer()
        self.theta = tf.Variable(initial_value=theta_init(shape=(input_shape[-1], self.units),
                                                          dtype='float32'),
                                 trainable=True)

        ### Build Bias weights
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=bias_init(shape=(self.units,),
                                                        dtype='float32'),
                                trainable=True)

        ### Build delta
        delta_coef_matrix = tf.Variable(tf.random_normal([self.units, self.number_of_quantiles]))
        delta_0_matrix = tf.Variable(tf.random_normal([1, self.number_of_quantiles]))
        delta_matrix = tf.concat([delta_0_matrix, delta_coef_matrix], axis=0)

        self.beta_matrix = tf.transpose(tf.cumsum(tf.transpose(delta_matrix)))

        delta_vec = delta_matrix[1:self.units, 1:self.number_of_quantiles]
        delta_0_vec = delta_matrix[0, 1:self.number_of_quantiles]
        delta_minus_vec = tf.maximum(0, -delta_vec)
        self.delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, 0)
        self.delta_0_vec_clipped = tf.clip_by_value(delta_0_vec,
                                                    clip_value_min=tf.reshape(self.delta_minus_vec_sum,
                                                                              np.shape(delta_0_vec)),
                                                    clip_value_max=tf.convert_to_tensor(
                                                        (np.ones(np.shape(delta_0_vec)) * np.inf)))

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.theta) + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def objective_function(X, y, quantiles_tf, quantiles, penalty, lambda_obj, input, input_dim, layer_1_obj, layer_2_obj, output_obj):
    output_y_tiled = tf.tile(y, np.shape(len(quantiles),1))

    delta_constraint = output_obj.delta_0_vec_clipped - output_obj.delta_minus_vec_sum
    delta_clipped = tf.clip_by_value(delta_constraint, clip_value_min = 10^^(-20), clip_value_max = np.inf)

    predicted_y_no_penalty = tf.matmul(input, output_obj.beta_matrix[1:(input_dim + 1)])
    predicted_y_modified = predicted_y_no_penalty + tf.cumsum( tf.concat([output_obj.beta_matrix[1, 1],
                                                               output_obj.delta_0_vec_clipped], axis=1),
                                                               axis=1)
    predicted_y = tf.matmul(input, output_obj.beta_matrix[1:(input_dim + 1), :]) + output_obj.beta_matrix[1, :]

    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), np.shape(output_y_tiled))

    diff_y = output_y_tiled - predicted_y_tiled
    quantile_loss = tf.reduce_mean(diff_y * (quantiles_tf - (tf.sign(-diff_y)+1)/2))

    objective_function_value = quantile_loss + penalty * (tf.reduce_mean(layer_1_obj.theta **2) \
                                + tf.reduce_mean(layer_2_obj.theta**2) \
                                + tf.reduce_mean(input.delta_coef_mat**2)) \
                                + lambda_obj * tf.reduce_mean(tf.abs(input.delta_0_vec - input.delta_0_vec_clipped))

    return objective_function_value

def optimize_l1_NMQN_RMSProp(X_train, y_train, lambda_objective_function, learning_rate,  max_deep_iter):
    #####  Compile keras model
    model.compile(optimizer=tf.keras.optimizers.rmsprop(learning_rate=learning_rate),  # default='rmsprop', an algorithm to be used in backpropagation
                  loss=lambda_objective_function, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['Accuracy'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
                  steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  )

    ##### Fit keras model on the dataset
    model.fit(X_train,  # input data
              y_train,  # target data
              # batch_size=30, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=max_deep_iter, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              validation_split=0.0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
              # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=False,  # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              # class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
              # sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
              initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
              steps_per_epoch=None,   # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
              workers=4,      # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=True,      # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )

    return model

def main():
    quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
    input, true_quantiles = # simulate data
    X_train, X_test, quantiles_in_sample, quantiles_out_sample = train_test_split(input, true_quantiles, test_size = 0.2, random_state = 0)

    tf_loss = lambda x, y: objective_function()
if __name__ == "__main__":
    main()
# Define the quantiles we’d like to estimate
quantile =  0.01
input, true_quantiles = data_simulation.simulate_gaussian_ar_garch(500, quantile, 0.9, 0.7, 0.25)

tf_loss = lambda x, y: loss_function(x, y, quantile)

X_train, X_test, quantiles_in_sample, quantiles_out_sample = train_test_split(input, true_quantiles,  test_size=0.2, random_state=0)

print(X_train[:5])

model = QRNN(input, input, objectivename='QRNN_model')
#####  Compile keras model
model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
              loss=tf_loss, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=['Accuracy'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
              steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )

##### Fit keras model on the dataset
model.fit(X_train, # input data
          X_train, # target data
          #batch_size=30, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=100, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          validation_split=0.0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          #class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          #sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=10, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
          workers=4, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=True, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
         )

print("Evaluate")
result = model.evaluate(X_train, X_train)
print(list(zip(model.metrics_names, result)))
print(model.predict(X_test))
print(quantiles_out_sample)

qrnn_plots.plot_results(X_test, quantiles_out_sample, model.predict(X_test), quantile)