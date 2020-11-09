### Parameters
# K_orig : the number of Tx antennas
# N_orig : the number of Rx antennas
# L : the number of layer
# train_iter = 0    : Test the performance of a trained model in test phase.
# train_iter > 0    : Train the neural networks

import tensorflow as tf
import numpy as np
import function as func

save_file_name = "trained_parameters"
detector_name = "ADMM"       # Types of MIMO detector : ZF, ADMM, Gated_ADMM

mod = 4     # BPSK(0) and QAM(4, 16, 64)

K_orig = 30      # the number of Tx antennas
N_orig = 60      # the number of Rx antennas

L = 30

train_iter = 0
train_batch_size = 2000
test_iter = 10
test_batch_size = 1000

snrdb_low = 8.0     # the lower bound of noise db
snrdb_high = 13.0   # the higher bound of noise db
num_snr = int(snrdb_high - snrdb_low + 1)
snr_low = 10.0 ** (snrdb_low / 10.0)
snr_high = 10.0 ** (snrdb_high / 10.0)

startingLearningRate = 0.0001   # the initial step size of the gradient descent algorithm
decay_factor = 0.97
decay_step_size = 1000

# initialize momentum parameter gamma
init_gamma = np.ones([L, 1])
for i in range (0,L):
    init_gamma[i] = i/(i+5)

# variable for the simulation result
sers = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_sers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
tmp_ser_iter = np.zeros([L, test_iter])
layer_ser_mean = np.zeros([L, num_snr])