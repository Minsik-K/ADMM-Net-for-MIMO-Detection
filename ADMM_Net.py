from Parameters import *

def ADMM_Net(batch_size, K, N, X, Y, H, H_T):

    x = tf.zeros([batch_size, K, 1])
    e = tf.zeros([batch_size, N, 1])

    v_hat = tf.zeros([batch_size, N, 1])
    v_pre = tf.zeros([batch_size, N, 1])

    # Learnable parameters
    if train_iter == 0:  # Load trained parameters
        parameters = np.load(save_file_name + '/ADMM_Net_parameters_('+str(K_orig)+'x'+str(N_orig)+' '+ str(mod)+'QAM MIMO).npy')

        delta = parameters[0]
        theta = parameters[1]
        alpha = parameters[2]
        beta = parameters[3]
        gamma = parameters[4]

    if train_iter > 0:  # Initialize learnable parameters
        delta = tf.Variable(tf.ones(L, 1) * 0.001)
        theta = tf.Variable(tf.ones(L, 1) * 1.5)
        alpha = tf.Variable(tf.ones(L, 1) * 1.5)
        beta = tf.Variable(tf.ones(L, 1) * 0.1)
        gamma = tf.cast(tf.Variable(tf.squeeze(init_gamma)),dtype=tf.float32)

    layer_ser = []
    layer_loss = []
    # The architecture of ADMM Net
    for i in range(0, L):

        # Initialize x = pseudo_inverse(H) * y
        # In lower order modulation, ADMM Net achieves good performance without initializing x.
        if i == 0:
            pinv_H = tf.matmul(tf.linalg.inv(tf.matmul(H_T, H)), H_T)
            x = tf.matmul(pinv_H, Y)

        x = x - (delta[i] * tf.matmul(H_T, e + (1 - 2 * beta[i]) * v_hat))
        x = func.demod_nonlinear(x, theta[i], mod)

        e = tf.matmul(H, x) - Y

        v = alpha[i] * e + (1 - alpha[i] * beta[i]) * v_hat
        v_hat = v + gamma[i] * (v - v_pre)

        v_pre = v

        # Nonlinear operator for demodulation
        x_hat = func.demod_nonlinear(x, 100, mod)
        x_result = func.demod_projection(x_hat, mod)

        # Calculate loss and SER
        LOSS, SER = func.loss(X, x_hat, x_result, K, mod)

        layer_loss.append(LOSS)
        layer_ser.append(SER)

    return x_hat, x_result, layer_loss, layer_ser, delta, theta, alpha, beta, gamma