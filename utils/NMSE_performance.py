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
def denorm_H4(data,minmax_file,link_type='down', timeslot=0):
    fieldnames = ['link','min','max']
    if "csv" in minmax_file:
        with open(minmax_file) as csv_file:
            csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
            for row in csv_reader:
                if row['link']==link_type:
                    d_min = float(row['min'])
                    d_max = float(row['max'])
    elif "pkl" in minmax_file:
        with open(f"{minmax_file}", "rb") as f:
            extrema_dict = pickle.load(f)
            f.close()
        extrema = extrema_dict[f"H_{link_type}_ext"]
        if timeslot != -1:
            d_min, d_max = np.min(extrema[0]), np.max(extrema[1]) # assume single timeslot performance
        else:
            d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    data = (data+1)/2*(d_max-d_min)+d_min
    return data

### helper function: normalize H4 (minmax scaling)
def renorm_H4(data, minmax_file, link_type='down', timeslot=0):
    fieldnames = ['link','min','max']
    if "csv" in minmax_file:
        with open(minmax_file) as csv_file:
            csv_reader = csv.DictReader(csv_file,fieldnames=fieldnames,delimiter=',')
            for row in csv_reader:
                if row['link']==link_type:
                    d_min = float(row['min'])
                    d_max = float(row['max'])
    elif "pkl" in minmax_file:
        with open(f"{minmax_file}", "rb") as f:
            extrema_dict = pickle.load(f)
            f.close()
        extrema = extrema_dict[f"H_{link_type}_ext"]
        if timeslot != -1:
            d_min, d_max = np.min(extrema[0]), np.max(extrema[1]) # assume single timeslot performance
        else:
            d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    data = 2 * (data-d_min)/(d_max-d_min) - 1
    return data

### helper function: normalize tanh 
def renorm_tanh(data, meanvar_file, n_stddev=4):
    f = sio.loadmat(meanvar_file)
    mu, sigma = f["mean_all"], np.sqrt(f["var_all"])
    data = np.tanh((data-mu)/(n_stddev*sigma))
    return data

### helper function: normalize tanh 
def denorm_tanh(data, meanvar_file, n_stddev=4, eps=0.01):
    f = sio.loadmat(meanvar_file)
    mu, sigma = f["mean_all"], np.sqrt(f["var_all"])
    # data = np.tanh((data-mu)/(n_stddev*sigma))
    data = np.clip(data, -1+eps, 1-eps)
    data = np.arctanh(data)*n_stddev*sigma + mu
    return data

### helper function: denormalize H4 with spherical normalization
def denorm_sphH4(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    # iterate through batches, denormalize based on extrema of given timeslot
    with open(f"{minmax_file}", "rb") as f:
        extrema_dict = pickle.load(f)
        f.close()
    extrema = extrema_dict[f"H_{link_type}_ext"]
    if timeslot != -1:
        # print(f"--- Denorm by timeslot={timeslot} ---")
        d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    else:
        d_min = extrema[0][0]
        d_max = extrema[1][0]
    data = (data+1)/2*(d_max-d_min)+d_min
    # with open(f"{t1_power_file}1.pkl", "rb") as f:
    with open(f"{t1_power_file}.pkl", "rb") as f:
        batch_power = pickle.load(f)
        f.close()
    link_power = batch_power[f"pow_{link_type}"]
    temp = np.reshape(data, (data.shape[0], -1)) * link_power[:, None]
    data = np.reshape(temp, data.shape)
    return data

### helper function: renormalize H4 with spherical normalization
def renorm_sphH4(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    # with open(f"{t1_power_file}1.pkl", "rb") as f:
    with open(f"{t1_power_file}.pkl", "rb") as f:
        batch_power = pickle.load(f)
        f.close()
    link_power = batch_power[f"pow_{link_type}"]
    data = np.reshape(np.reshape(data, (data.shape[0], -1)) / link_power[:,None], data.shape)
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

### helper function: denormalize H3 [0,1] with spherical normalization, magnitude/phase
def denorm_sph_magH3(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    # undo minmax scaling on magnitude estimates
    with open(f"{minmax_file}", "rb") as f:
        extrema_dict = pickle.load(f)
        f.close()
    extrema = extrema_dict[f"H_{link_type}_ext"]
    if timeslot != -1:
        d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    else:
        d_min = np.min(extrema[0])
        d_max = np.max(extrema[1])
    if len(data.shape) == 4:
        data_mag, data_pha = data[:,0,:,:], data[:,1,:,:]
        concat_axis = 1
    elif len(data.shape) == 5:
        data_mag, data_pha = data[:,:,0,:,:], data[:,:,1,:,:]
        concat_axis = 2
    else:
        print("--- renorm_sph_magH3: data are not correct shape. Expected 4 or 5 axes. ---")
        return None
    # data_mag = np.clip(data_mag, 0, 1) # trying this; avoid negative magnitude values 
    data_mag = data_mag*(d_max-d_min) + d_min
    # incoming data channels are [mag, phase]; convert to re/im
    data_re = np.expand_dims(data_mag*np.cos(data_pha), axis=concat_axis)
    data_im = np.expand_dims(data_mag*np.sin(data_pha), axis=concat_axis)
    data = np.concatenate((data_re, data_im), axis=concat_axis)
    # denorm by sample power
    with open(f"{t1_power_file}.pkl", "rb") as f:
        batch_power = pickle.load(f)
        f.close()
    link_power = batch_power[f"pow_{link_type}"]
    # data = np.reshape(np.reshape(data, (data.shape[0], -1)) / link_power[:,None], data.shape)
    temp = np.reshape(data, (data.shape[0], -1)) * link_power[:, None]
    data = np.reshape(temp, data.shape)
    return data

### helper function: renormalize H3 [0,1] with spherical normalization, magnitude/phase
def renorm_sph_magH3(data, minmax_file, t1_power_file, batch_num, link_type='down', timeslot=0):
    with open(f"{t1_power_file}.pkl", "rb") as f:
        batch_power = pickle.load(f)
        f.close()
    link_power = batch_power[f"pow_{link_type}"]
    data = np.reshape(np.reshape(data, (data.shape[0], -1)) / link_power[:,None], data.shape)
    # split mag/phase
    if len(data.shape) == 4:
        data = data[:,0,:,:]+data[:,1,:,:]*1j
        concat_axis = 1
    elif len(data.shape) == 5:
        data = data[:,:,0,:,:]+data[:,:,1,:,:]*1j
        concat_axis = 2
    else:
        print("--- renorm_sph_magH3: data are not correct shape. Expected 4 or 5 axes. ---")
        return None
    data_mag = np.abs(data)
    data_ang = np.angle(data)
    # perform minmax scaling on magnitude estimates
    with open(f"{minmax_file}", "rb") as f:
        extrema_dict = pickle.load(f)
        f.close()
    extrema = extrema_dict[f"H_{link_type}_ext"]
    if timeslot != -1:
        d_min, d_max = extrema[0][timeslot], extrema[1][timeslot] # assume single timeslot performance
    else:
        d_min = np.min(extrema[0])
        d_max = np.max(extrema[1])
    data_mag = (data_mag-d_min)/(d_max-d_min)
    data = np.concatenate([np.expand_dims(data_mag, concat_axis), np.expand_dims(data_ang, concat_axis)], axis=concat_axis)
    return data

# calculate NMSE
def calc_NMSE(x_hat,x_test,T=3,pow_diff=None):
    results = {
                "mse_truncated": [0]*T,
                "nmse_truncated": [0]*T,
                "mse_full": [0]*T,
                "nmse_full": [0]*T
              }
    if T == 1:
        x_test_temp =  np.reshape(x_test, (len(x_test), -1))
        x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
        pow_diff = np.reshape(pow_diff, (pow_diff.shape[0],1)) if type(pow_diff) != type(None) else None
    else:
        x_test_temp =  np.reshape(x_test, (x_test.shape[0]*x_test.shape[1], -1))
        x_hat_temp =  np.reshape(x_hat, (x_hat.shape[0]*x_hat.shape[1], -1))
        pow_diff_temp = np.reshape(pow_diff, (pow_diff.shape[0]*pow_diff.shape[1],)) if type(pow_diff) != type(None) else None
    power = np.sum(abs(x_test_temp)**2, axis=1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
    temp = mse/power
    mse = mse[np.nonzero(power)] 
    pow_diff_temp = pow_diff_temp[np.nonzero(power)] if type(pow_diff) != type(None) else None
    power = power[np.nonzero(power)] 
    temp = mse[np.nonzero(power)] / power[np.nonzero(power)]
    results["avg_mse_truncated"] = np.mean(mse)
    results["avg_nmse_truncated"] = 10*math.log10(np.mean(temp))
    print(f"Average Truncated | NMSE = {results['avg_nmse_truncated']} | MSE = {results['avg_mse_truncated']:.4E}")
    if type(pow_diff) != type(None):
        temp = (mse + pow_diff_temp) / (power + pow_diff_temp)
        results["avg_mse_full"] = np.mean(mse + pow_diff_temp)
        results["avg_nmse_full"] = 10*math.log10(np.mean(temp))
        print(f"Average Full | NMSE = {results['avg_nmse_full']} | MSE = {results['avg_mse_full']:.4E}")
    if T != 1:
        for t in range(T):
            x_test_temp =  np.reshape(x_test[:, t, :, :, :], (len(x_test[:, t, :, :, :]), -1))
            x_hat_temp =  np.reshape(x_hat[:, t, :, :, :], (len(x_hat[:, t, :, :, :]), -1))
            pow_diff_temp = np.squeeze(pow_diff[:,t,:]) if type(pow_diff) != type(None) else None
            power = np.sum(abs(x_test_temp)**2, axis=1)
            mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
            mse = mse[np.nonzero(power)] 
            pow_diff_temp = pow_diff_temp[np.nonzero(power)] if type(pow_diff) != type(None) else None
            power = power[np.nonzero(power)] 
            temp = mse/power
            results["mse_truncated"][t] = np.mean(mse) 
            results["nmse_truncated"][t] = 10*math.log10(np.mean(temp))
            if type(pow_diff) != type(None):
                temp = (mse + pow_diff_temp) / (power + pow_diff_temp)
                results["mse_full"][t] = np.mean(mse) 
                results["nmse_full"][t] = 10*math.log10(np.mean(temp))
                print(f"t{t+1} | Truncated NMSE = {results['nmse_truncated'][t]} | Full NMSE = {results['nmse_full'][t]}")
            else:
                print(f"t{t+1} | Truncated NMSE = {results['nmse_truncated'][t]}")
    return results

# method 1: return NMSE for single timeslot
# def get_NMSE(x_hat, x_test, n_del=32, n_ang=32, pow_diff_timeslot=None, return_mse=False):
#     """ return NMSE in dB. optionally return MSE. """
#     x_test_temp =  np.reshape(x_test, (len(x_test), -1))
#     x_hat_temp =  np.reshape(x_hat, (len(x_hat), -1))
#     power = np.sum(abs(x_test_temp)**2, axis=1) # shape = (N, img_total) -> (N,1)
#     pow_diff = np.zeros(power.shape) if type(pow_diff_timeslot) == type(None) else pow_diff_timeslot
#     mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1) + pow_diff # (N,1)
#     # mse = mse[np.nonzero(power)] 
#     # power = power[np.nonzero(power)] 
#     # print(f"x_test_temp.shape: {x_test_temp.shape} - power.shape: {power.shape} - mse.shape: {mse.shape}")
#     nmse =  10*math.log10(np.mean(mse/(power+pow_diff))) # np.mean((N,1) / (N,1)) 
#     if return_mse:
#         return [np.mean(mse), nmse]
#     else:
#         return nmse

# method 2: use trace of error matrix
def get_NMSE(x_hat, x_test, n_del=32, n_ang=32, return_mse=False, pow_diff_timeslot=None, n_train=0):
    """
    return NMSE in dB. optionally return MSE
    reshape based on delay/angle counts
    """
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
    N_zero = 0
    for i in range(N):
        pow_diff = 0 if type(pow_diff_timeslot) == type(None) else pow_diff_timeslot[i]
        tr_term = np.trace(np.matmul(x_err[i,:,:],np.conj(x_err[i,:,:].T)))
        if np.real(power[i]) > 0: # ignore term if power is 0
            mse += (tr_term + np.real(pow_diff)) / N
            nmse += (tr_term + np.real(pow_diff)) / ((np.real(power[i]) + np.real(pow_diff))*N)
        else:
            N_zero += 1
    nmse =  10*math.log10(np.real(nmse))
    # print(f"--- N_zero={N_zero} ---")

    if return_mse:
        return [np.real(mse).item(), nmse]
    else:
        return nmse

# calculate rms for a window of a signal
def calc_rms(x, window, idx):
    temp = np.mean(x[idx:idx+window]*np.conj(x[idx:idx+window]))
    return np.sqrt(temp)
