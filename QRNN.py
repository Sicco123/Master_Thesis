import tensorflow as tf
from tensorflow import keras
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

class QRNN(tf.keras.Model):

  def __init__(self,input, output, **kwargs):
    super().__init__()
    self.layer_1 = Flatten()
    self.layer_2 = Dense(units=15, activation = "sigmoid") # Choose correct number of units
    self.layer_3 = Dense(units=1) # Choose correct activation function
    # possible add "Dropout" to prevent overfitting

  def call(self, inputs):
    x = self.layer_1(inputs)
    x = self.layer_2(x)
    x = self.layer_3(x)
    return x

def loss_function(y, output, quantile):
    error = tf.subtract(y, output)
    loss = tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error), axis=-1)
    #print(loss)
    return loss

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