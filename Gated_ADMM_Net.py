from Parameters import *

def Gated_ADMM_Net(batch_size, K, N, X, Y, H, H_T):

    x = tf.zeros([batch_size, K, 1])
    e = tf.zeros([batch_size, N, 1])

    v_hat = tf.zeros([batch_size, N, 1])
    v_pre = tf.zeros([batch_size, N, 1])

    c = tf.ones([tf.shape(x)[0], 1, 1])


    # Learnable parameters
    if train_iter == 0:   # Load trained parameters
        Gated_ADMM_Net_parameters_d_t_b = np.load(save_file_name+'/Gated_ADMM_Net_parameters_d_t_b_('
                                                  +str(K_orig)+'x'+str(N_orig)+' '+ str(mod)+'QAM MIMO).npy')
        Gated_ADMM_Net_parameters_W1_W2 = np.load(save_file_name+'/Gated_ADMM_Net_parameters_W1_W2_('
                                                  +str(K_orig)+'x'+str(N_orig)+' '+ str(mod)+'QAM MIMO).npy')
        Gated_ADMM_Net_parameters_b1_b2 = np.load(save_file_name+'/Gated_ADMM_Net_parameters_b1_b2_('
                                                  +str(K_orig)+'x'+str(N_orig)+' '+ str(mod)+'QAM MIMO).npy')

        delta = Gated_ADMM_Net_parameters_d_t_b[0]
        theta = Gated_ADMM_Net_parameters_d_t_b[1]
        beta = Gated_ADMM_Net_parameters_d_t_b[2]
        W1 = Gated_ADMM_Net_parameters_W1_W2[0]
        W2 = Gated_ADMM_Net_parameters_W1_W2[1]
        b1 = Gated_ADMM_Net_parameters_b1_b2[0]
        b2 = Gated_ADMM_Net_parameters_b1_b2[1]

    if train_iter > 0:  # Initialize learnable parameters
        delta = tf.Variable(tf.ones(L, 1) * 0.0001)
        theta = tf.Variable(tf.ones(L, 1) * 1.5)
        beta = tf.Variable(tf.ones(L, 1) * 0.1)
        W1 = tf.Variable(tf.ones([L, 1, K + N]) * 0.001)
        b1 = tf.Variable(tf.ones([L, 1, 1]) * 0.001)
        W2 = tf.Variable(tf.ones([L, 1, K + N]) * 0.001)
        b2 = tf.Variable(tf.ones([L, 1, 1]) * 0.001)

    layer_ser = []
    layer_loss = []
    # The architecture of Gated ADMM Net
    for i in range(0, L):

        # initialize X = pseudo_inverse(H) * y
        if i == 0:
            pinv_H = tf.matmul(tf.linalg.inv(tf.matmul(H_T, H)), H_T)
            x = tf.matmul(pinv_H, Y)

        x = x - (delta[i] * tf.matmul(H_T, e + (1 - 2 * beta[i]) * v_hat))
        x = func.demod_nonlinear(x, theta[i], mod)

        e = tf.matmul(H, x) - Y

        state = tf.concat([x, v_hat], 1)
        alpha = tf.matmul(c * (W1[i, :, :]), state) + b1[i, :, :]
        gamma = tf.matmul(c * (W2[i, :, :]), state) + b2[i, :, :]
        alpha = 2 * tf.sigmoid(alpha)
        gamma = tf.sigmoid(gamma)

        v = alpha * e + (1 - alpha * beta[i]) * v_hat
        v_hat = v + gamma * (v - v_pre)

        v_pre = v

        # Nonlinear operator for demodulation
        x_hat = func.demod_nonlinear(x, 100, mod)
        x_result = func.demod_projection(x_hat, mod)

        # Calculate loss and SER
        LOSS, SER = func.loss(X, x_hat, x_result, K, mod)

        layer_loss.append(LOSS)
        layer_ser.append(SER)

    return x_hat, x_result, layer_loss, layer_ser, delta, theta, beta, W1, W2, b1, b2