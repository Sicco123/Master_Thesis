import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import data_simulation
tf.disable_v2_behavior()

def l1_p(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, lambda_obj, penalty = 0):
    input_dim = len(X[0])
    n = len(X)
    r = len(tau)
    p = hidden_dim2 + 1
    tau_mat = np.expand_dims(np.repeat(np.transpose(tau), [n]), axis=1) # repeat tau n times

    input_x = tf.placeholder(tf.float32, [None, input_dim])#shape)
    output_y = tf.placeholder(tf.float32,[None, 1]) #shape)
    output_y_tiled = tf.tile(output_y, [r, 1])#shape)
    tau_tf = tf.placeholder(tf.float32, [n*r, 1]) #shape)

    ### layer_1
    hidden_theta_1 = tf.Variable(tf.random_normal([input_dim, hidden_dim1]))
    hidden_bias_1 = tf.Variable(tf.random_normal([hidden_dim1]))
    hidden_layer_1 = tf.nn.sigmoid(tf.matmul(input_x, hidden_theta_1) + hidden_bias_1)

    ### layer_2
    hidden_theta_2 = tf.Variable(tf.random_normal([hidden_dim1, hidden_dim2]))
    hidden_bias_2 = tf.Variable(tf.random_normal([hidden_dim2]))
    feature_vec = tf.nn.sigmoid(tf.matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2)

    ### output layer
    delta_coef_mat = tf.Variable(tf.random_normal([hidden_dim2, r]))
    delta_0_mat = tf.Variable(tf.random_normal([1,r]))

    delta_mat = tf.concat([delta_0_mat, delta_coef_mat], axis = 0 )
    beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat)))

    delta_vec = delta_mat[1:, 1:]
    delta_0_vec = delta_mat[0:1, 1:]
    delta_minus_vec = tf.maximum(0.0, -delta_vec)
    delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis = 0)
    delta_0_vec_clipped = tf.clip_by_value(delta_0_vec, clip_value_min = tf.reshape(delta_minus_vec_sum, delta_0_vec.shape), clip_value_max = tf.convert_to_tensor(
                                                        (np.ones(np.shape(delta_0_vec)) * np.inf), dtype = 'float32'))

    #### optimization
    delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
    delta_clipped = tf.clip_by_value(delta_constraint, clip_value_min = 10**(-20), clip_value_max = np.inf)

    predicted_y_modified = tf.matmul(feature_vec, beta_mat[1:p, :]) + tf.cumsum(tf.concat([beta_mat[1:2, 1:2],
                                                                             delta_0_vec_clipped], axis=1),
                                                                  axis=1)

    predicted_y = tf.matmul(feature_vec, beta_mat[1:p, :]) + beta_mat[1,:]
    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y),  [n*r,1] )

    diff_y = output_y_tiled - predicted_y_tiled
    quantile_loss = tf.reduce_mean(diff_y * (tau_tf - (tf.sign(-diff_y)+1)/2))

    objective_fun = quantile_loss + penalty * (tf.reduce_mean(hidden_theta_1**2)+tf.reduce_mean(hidden_theta_2**2)+tf.reduce_mean(delta_coef_mat**2))+ lambda_obj * tf.reduce_mean(tf.abs(delta_0_vec - delta_0_vec_clipped))

    train_opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(objective_fun)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #tmp_vec = ...

    for step in range(max_deep_iter):
        sess.run(train_opt, feed_dict = {input_x: X, output_y: y ,tau_tf: tau_mat})

    y_predict = sess.run(predicted_y_modified, feed_dict = {input_x : X})
    y_test_predict = sess.run(predicted_y_modified, feed_dict = {input_x : test_X})
    y_valid_predict = sess.run(predicted_y_modified, feed_dict = {input_x : valid_X})
    sess.close()
    barrier_result = [y_predict,  y_valid_predict, y_test_predict]
    return(barrier_result)

### train data
n = 1000
input_dim = 1
x_data, y_data = data_simulation.moon_data(n, input_dim)

### valid data
x_valid_data, y_valid_data = data_simulation.moon_data(n, input_dim)

### test data
x_test_data, y_test_data = data_simulation.moon_data(n, input_dim)

### Model fitting
tau_vec = np.arange(0.1, 1, 0.1)

fit_result  = l1_p(X = x_data,
                   y = y_data,
                   test_X = x_test_data,
                   valid_X = x_valid_data,
                   tau = tau_vec,
                   hidden_dim1 = 4,
                   hidden_dim2 = 4,
                   learning_rate = 0.005,
                   max_deep_iter = 2000,
                   penalty = 0,
                   lambda_obj = 5)

plt.scatter(x_data , y_data)
for i in range(len(fit_result[2][0])):
    plt.scatter(x_data, fit_result[0][:,i])

plt.show()