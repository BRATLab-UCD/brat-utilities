# NMSE_performance.py
from .unpack_json import *
import scipy.io as sio
import numpy as np
import math
import time
import sys
import csv

### load model with weights: supersedes same func in unpack_json
def model_with_weights(model_dir,network_name,weights_bool=True):
    # load json and create model
    outfile = "{}/{}.json".format(model_dir,network_name)
    json_file = open(outfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print('-> Loading model from {}...'.format(outfile))
    model = model_from_json(loaded_model_json)
    # load weights outto new model
    outfile = "{}/{}.h5".format(model_dir,network_name)
    print('-> Loading weights from {}...'.format(outfile))
    model.load_weights(outfile)
    return model

def predict_with_model(model,d,aux_bool=0):
    #Testing data
    tStart = time.time()
    print('-> Predicting on test set...')
    if aux_bool:
        x_hat = model.predict([d['aux_test'],d["test"]])
    else:
        x_hat = model.predict(d["test"])
    tEnd = time.time()
    print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/d["test"].shape[0],d["test"].shape[0]))
    return x_hat

### helper function: denormalize H3
def denorm_H3(data,minmax_file,link_type='down'):
    fieldnames = ['link','min','max']
    with open(minmax_file) as csv_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
        for row in csv_reader:
            if row['link']==link_type:
                d_min = float(row['min'])
                d_max = float(row['max'])
    data = data*(d_max-d_min)+d_min
    return data

### helper function: denormalize H4
def denorm_H4(data,minmax_file,link_type='down'):
    fieldnames = ['link','min','max']
    with open(minmax_file) as csv_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
        for row in csv_reader:
            if row['link']==link_type:
                d_min = float(row['min'])
                d_max = float(row['max'])
    data = (data+1)/2*(d_max-d_min)+d_min
    return data

# calculate NMSE
def calc_NMSE(x_hat,x_test,T=3):
    if T == 1:
        x_test_temp =  np.reshape(x_test, (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
    else:
        x_test_temp =  np.reshape(x_test[:, :, :, :, :], (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat[:, :, :, :, :], (len(x_hat), -1))
    power = np.sum(abs(x_test_temp)**2, axis=1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
    mse = mse[np.nonzero(power)] 
    power = power[np.nonzero(power)] 
    temp = mse/power
    print("Overall NMSE is {}".format(10*math.log10(np.mean(temp))))
    if T != 1:
        for t in range(T):
            x_test_temp =  np.reshape(x_test[:, t, :, :, :], (len(x_test[:, t, :, :, :]), -1))
            x_hat_temp =  np.reshape(x_hat[:, t, :, :, :], (len(x_hat[:, t, :, :, :]), -1))
            power = np.sum(abs(x_test_temp)**2, axis=1)
            mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
            mse = mse[np.nonzero(power)] 
            power = power[np.nonzero(power)] 
            temp = mse/power
            print("NMSE at t{} is {}".format(t+1, 10*math.log10(np.mean(temp))))

# method 1: return NMSE for single timeslot
def get_NMSE(x_hat, x_test, return_mse=False):
    """ return NMSE in dB. optionally return MSE. """
    x_test_temp =  np.reshape(x_test, (len(x_test), -1))
    x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
    power = np.sum(abs(x_test_temp)**2, axis=1) # shape = (N, img_total) -> (N,1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1) # (N,1)
    # mse = mse[np.nonzero(power)] 
    # power = power[np.nonzero(power)] 
    # print(f"x_test_temp.shape: {x_test_temp.shape} - power.shape: {power.shape} - mse.shape: {mse.shape}")
    nmse =  10*math.log10(np.mean(mse/power)) # np.mean((N,1) / (N,1)) 
    if return_mse:
        return [np.mean(mse), nmse]
    else:
        return nmse

# method 2: use trace of error matrix
# def get_NMSE(x_hat, x_test, n_del=32, n_ang=32, return_mse=False):
#     """
#     return NMSE in dB. optionally return MSE
#     reshape based on delay/angle counts
#     """
#     x_err = np.reshape(x_test - x_hat, (x_test.shape[0], n_del, n_ang))
#     x_test = np.reshape(x_test, (len(x_test), -1))
#     power = np.sum(np.conj(x_test)*x_test, axis=1)
#     print(f"x_err.shape: {x_err.shape} - power.shape: {power.shape}")
#     # mse = np.sum(np.conj(x_err)*x_err, axis=1)
#     # mse = mse[np.nonzero(power)]
#     # power = power[np.nonzero(power)] 
#     mse = 0
#     nmse = 0
#     N = x_err.shape[0]
#     for i in range(N):
#         tr_term = np.trace(np.matmul(x_err[i,:,:],np.conj(x_err[i,:,:].T)))
#         mse += tr_term / N
#         nmse += tr_term / (np.real(power[i])*N)
#     nmse =  10*math.log10(nmse)
#     if return_mse:
#         return [np.real(np.mean(mse)), nmse]
#     else:
#         return nmse

# calculate rms for a window of a signal
def calc_rms(x, window, idx):
    temp = np.mean(x[idx:idx+window]*np.conj(x[idx:idx+window]))
    return np.sqrt(temp)