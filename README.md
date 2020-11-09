# ADMM Network for MIMO Detection
 
**This repository contains the and hyperparameters code for the paper:**

*M. Kim and D. Park, "Learnable MIMO Detection Networks based on Inexact ADMM," accepted for publication in IEEE Transactions on Wireless Communications, 2020.* 
<!--
(https://...)
-->

If you use this code for your research, please cite our papers.

## Software Versions
* python 3.5.6
* tensorflow-gpu 1.8.0
* numpy 1.16.4

## ADMM Net train/test
* Select the MIMO system paramters in ```parameter.py```.
* Run the ```main.py``` file.
* The latest traing parameters will be saved in ```./trained_parameters```.
* The test results will be printed at console.
* If you have trained parameters in ```./trained_parameters``` and want just a test, set ```train_iter = 0``` in ```parameter.py```.

## Data set
* Data set is heavily based on Neev Samuel's implementation of a DetNet (https://github.com/neevsamuel).
