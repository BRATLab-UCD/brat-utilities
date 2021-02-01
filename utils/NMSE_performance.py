# NMSE_performance.py
from .unpack_json import *
import scipy.io as sio
import numpy as np
import pickle
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

### helper function: normalize H4 (minmax scaling)
def renorm_H4(data, minmax_file, link_type='down'):
    fieldnames = ['link','min','max']
    with open(minmax_file) as csv_file:
        csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
        for row in csv_reader:
            if row['link']==link_type:
                d_min = float(row['min'])
                d_max = float(row['max'])
    data = 2 * (data-d_min)/(d_max-d_min) - 1
    return data

### helper function: denormalize H4 with spherical normalization
def denorm_sphH4(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    # iterate through batches, denormalize based on extrema of given timeslot
    with open(f"{minmax_file}", "rb") as f:
        extrema_dict = pickle.load(f)
        f.close()
    extrema = extrema_dict[f"H_{link_type}_ext"]
    if timeslot != -1:
        print(f"--- Denorm by timeslot={timeslot} ---")
        d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    else:
        # print(f"--- Denorm by merging timeslots ---")
        # d_min = np.min(extrema[0])
        # d_max = np.max(extrema[1])
        print(f"--- Denorm by t1 extrema ---")
        d_min = extrema[0][0]
        d_max = extrema[1][0]
    data = (data+1)/2*(d_max-d_min)+d_min
    # use t1 power to denormalize everything 
    for i in range(batch_num):
        with open(f"{t1_power_file}{i+1}.pkl", "rb") as f:
            batch_power = pickle.load(f)
            f.close()
        link_power = batch_power[f"pow_{link_type}"]
        if i == 0: # assuming batches are same size
            link_size = link_power.shape[0]
            batch_size = int(data.shape[0] / batch_num)
            # print(f"--- In denorm, detected batch_size={batch_size} ---")
        temp = np.reshape(data[i*batch_size:(i+1)*batch_size,:,:], (batch_size, -1))
        temp = temp * link_power[link_size-batch_size:,None]
        data[i*batch_size:(i+1)*batch_size,:,:] = np.reshape(temp, (batch_size,)+data.shape[1:])
    return data

### helper function: renormalize H4 with spherical normalization
def renorm_sphH4(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    # iterate through batches, normalize each timeslot by t1 power
    for i in range(batch_num):
        with open(f"{t1_power_file}{i+1}.pkl", "rb") as f:
            batch_power = pickle.load(f)
            f.close()
        link_power = batch_power[f"pow_{link_type}"]
        if i == 0: # assuming batches are same size
            link_size = link_power.shape[0]
            batch_size = int(data.shape[0] / batch_num)
            # print(f"--- In denorm, detected batch_size={batch_size} ---")
        data[i*batch_size:(i+1)*batch_size,:,:] = np.reshape(np.reshape(data[i*batch_size:(i+1)*batch_size,:,:], (batch_size, -1)) / link_power[link_size-batch_size:,None], (batch_size,)+data.shape[1:])
    # perform minmax scaling on estimates
    with open(f"{minmax_file}", "rb") as f:
        extrema_dict = pickle.load(f)
        f.close()
    extrema = extrema_dict[f"H_{link_type}_ext"]
    if timeslot != -1:
        d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    else:
        d_min = np.min(extrema[0])
        d_max = np.max(extrema[1])
    data = 2 * (data-d_min)/(d_max-d_min) - 1
    return data

# calculate NMSE
def calc_NMSE(x_hat,x_test,T=3,diff_test=None):
    results = {}
    if T == 1:
        x_test_temp =  np.reshape(x_test, (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
    else:
        x_test_temp =  np.reshape(x_test[:, :, :, :, :], (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat[:, :, :, :, :], (len(x_hat), -1))
    power = np.sum(abs(x_test_temp)**2, axis=1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
    temp = mse/power
    # mse = mse[np.nonzero(power)] 
    # power = power[np.nonzero(power)] 
    # temp = mse[np.nonzero(power)] / power[np.nonzero(power)]
    results["avg_truncated"] = 10*math.log10(np.mean(temp))
    print("Average Truncated NMSE is {}".format(results["avg_truncated"]))
    if type(diff_test) != type(None):
        power += diff_test
        mse += diff_test
        # temp = mse[np.nonzero(power)] / power[np.nonzero(power)]
        temp = mse/power
        results["avg_full"] = 10*math.log10(np.mean(temp))
        print("Average Full NMSE is {}".format(results["avg_full"]))
    # TODO: return results for all timeslots
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
    return results

# method 1: return NMSE for single timeslot
# def get_NMSE(x_hat, x_test, return_mse=False):
#     """ return NMSE in dB. optionally return MSE. """
#     x_test_temp =  np.reshape(x_test, (len(x_test), -1))
#     x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
#     power = np.sum(abs(x_test_temp)**2, axis=1) # shape = (N, img_total) -> (N,1)
#     mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1) # (N,1)
#     # mse = mse[np.nonzero(power)] 
#     # power = power[np.nonzero(power)] 
#     # print(f"x_test_temp.shape: {x_test_temp.shape} - power.shape: {power.shape} - mse.shape: {mse.shape}")
#     nmse =  10*math.log10(np.mean(mse/power)) # np.mean((N,1) / (N,1)) 
#     if return_mse:
#         return [np.mean(mse), nmse]
#     else:
#         return nmse

# method 2: use trace of error matrix
def get_NMSE(x_hat, x_test, n_del=32, n_ang=32, return_mse=False, pow_diff_timeslot=None):
    """
    return NMSE in dB. optionally return MSE
    reshape based on delay/angle counts
    """
    print(f"x_test.shape: {x_test.shape} - x_hat.shape: {x_hat.shape}")
    x_err = np.reshape(x_test - x_hat, (x_test.shape[0], n_ang, n_del))
    x_test = np.reshape(x_test, (len(x_test), -1))
    power = np.sum(np.conj(x_test)*x_test, axis=1)
    # print(f"x_err.shape: {x_err.shape} - power.shape: {power.shape}")
    # mse = np.sum(np.conj(x_err)*x_err, axis=1)
    # mse = mse[np.nonzero(power)]
    # power = power[np.nonzero(power)] 
    mse = 0
    nmse = 0
    N = x_err.shape[0]
    for i in range(N):
        pow_diff = 0 if type(pow_diff_timeslot) == type(None) else pow_diff_timeslot[i]
        tr_term = np.trace(np.matmul(x_err[i,:,:],np.conj(x_err[i,:,:].T)))
        mse += tr_term / N
        nmse += (tr_term + pow_diff) / ((np.real(power[i]) + pow_diff)*N)
    nmse =  10*math.log10(nmse)

    if return_mse:
        return [np.real(mse), nmse]
    else:
        return nmse

# calculate rms for a window of a signal
def calc_rms(x, window, idx):
    temp = np.mean(x[idx:idx+window]*np.conj(x[idx:idx+window]))
    return np.sqrt(temp)