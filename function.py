from Parameters import *

def loss(x_true, x_hat, x_result, K, mod):
    if mod == 0:
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_true - x_hat), 1))
        ser = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.not_equal(x_true, x_result), tf.float32), 1))

    else:
        X_c = tf.complex(x_true[:, 0 : tf.cast(K/2, dtype='int32')], x_true[:, tf.cast(K/2, dtype='int32') : K])
        x_c = tf.complex(x_result[:, 0 : tf.cast(K/2, dtype='int32')], x_result[:, tf.cast(K/2, dtype='int32') : K])

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_true - x_hat), 1))
        ser = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.not_equal(X_c, x_c), tf.float32), 1))

    return loss, ser

# Differentiable demodulation operator proposed in the paper
def demod_nonlinear(x, theta, mod):
    if mod == 0 or mod == 4:
        x_hat = tf.nn.tanh(theta * x)

    elif mod == 16:
        x_hat = tf.nn.tanh(theta*x) + tf.nn.tanh(theta*(x-2)) + tf.nn.tanh(theta*(x+2))

    if mod == 64:
        x_hat = tf.nn.tanh(x)+\
                tf.nn.tanh(theta*(x-2)) + tf.nn.tanh(theta*(x+2)) + \
                tf.nn.tanh(theta*(x-4)) + tf.nn.tanh(theta*(x+4)) + \
                tf.nn.tanh(theta*(x-6)) + tf.nn.tanh(theta*(x+6))

    return x_hat

def demod_projection(x, mod):
    if mod == 0:
        x_result = tf.sign(x)

    else:
        x_result = tf.clip_by_value(x, -(np.log2(mod)-1)-0.5, (np.log2(mod)-1)+0.5)
        x_result = (tf.round((x_result + (np.log2(mod)-1))/2) )*2 - (np.log2(mod)-1)

    return x_result