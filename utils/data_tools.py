# data_tools.py
# functions for importing/manipulating data for training/validation
import numpy as np
import h5py
import scipy.io as sio
from .unpack_json import get_keys_from_json
# from QuantizeData import quantize 

# TODO: write this, if useful
# def make_dummy_data(N, n_delay=32, n_angle=32, n_channels=2, data_format="channels_first"):
#     """ make dummy CSI data for proving out different functions """

def make_ar_data(data, p, n_chan=2, n_delay=32, n_angle=32, batch_factor=None, mode="matrix", stride=1, stack=0, backwards=True):
    """
    make time-series of p-length inputs and one-step ahead outputs for vector autoregression

    Parameters
    ----------
    data : array-like, shape (batch_num, T, k)
           full data (train, test, or val)
    p : int
        number of steps for VAR(p)
    n_chan: int
            number of channels in data matrix, typically real/imaginary (i.e., 2)
    n_delay: int
             num of delay values per angle in CSI matrix
    n_angle: int
             num of angular values per delay in CSI matrix
    batch_factor: int
                  divisor to reduce num of elements in CSI matrix
                  divides along batch_num axis of data
    mode: str
          different data types to return based on different linear model assumptions
          "matrix" -- elements in y are lin comb of all other elements in CSI matrices from p timeslots
          "scalar" -- elements in y are lin comb of elements at same (delay,angle) of CSI matrices from p timeslots
          "angular" -- elements in y are lin comb of elements across all angles of CSI matrices from p timeslots
          "angular_corr" -- same premise as 'angular' using empirical correlation matrices 
    stride: int
            increment of AR process steps; (H_0, H_{stride}, ... H_{stride*(p-1)}) to predict H_{stride*p}
    stack: int
           offset index of current stack -- use to grab different subsequences from the same sequence 
    backwards: bool
               If True, then predict last timeslot (y) based on previous p timeslots (Z)
               If False, then p+1-th timeslot (y) based on first p timeslots (Z)
    """

    img_total = n_chan*n_delay*n_angle
    T = data.shape[1]
    e_i = T-1-stack
    assert(stack+p*stride < T)
    # print(f"--- Z_idx: {[i for i in range(e_i-p*stride,e_i,stride)]} , y_idx: {e_i}---")
    # TODO: add stack to other modes
    # TODO: add end-indexing to other modes
    if mode == "matrix":
        Z = np.reshape(data[:,:p*stride:stride,:], (data.shape[0], p*img_total))
        y = data[:,p*stride,:]
    elif mode == "scalar":
        Z = data[:,:p*stride:stride,:]
        Z = np.reshape(np.transpose(Z, (0,2,1)), (data.shape[0]*img_total, p))
        y = data[:,p*stride,:]
        y = np.reshape(y, (data.shape[0]*img_total, 1))
    elif mode == "angular":
        Z = np.reshape(data[:,:p*stride:stride,:], (data.shape[0], p, n_chan, n_delay, n_angle))
        Z = np.reshape(np.transpose(Z, (0,3,1,2,4)), (data.shape[0]*n_delay, p*n_chan*n_angle))
        y = np.reshape(data[:,p*stride,:], (data.shape[0], n_chan, n_delay, n_angle))
        y = np.reshape(np.transpose(y, (0,2,1,3)), (data.shape[0]*n_delay, n_chan*n_angle))
        # elif n_chan == 0:
        #     Z = np.reshape(combine_complex(data[:,:p,:], n_delay, n_angle), (data.shape[0],p,n_delay,n_angle))
        #     Z = np.reshape(np.transpose(Z, (0,2,1,3)), (data.shape[0]*n_delay, p, 2*n_angle))
        #     y = np.reshape(combine_complex(np.expand_dims(data[:,p,:], axis=1), n_delay, n_angle), (data.shape[0]*n_delay,2*n_angle))
    elif mode == "angular_corr_vect":
        Z = np.reshape(combine_complex(data[:,stack:stack+p*stride:stride,:], n_delay, n_angle), (data.shape[0],p,n_delay,n_angle))
        Z = np.reshape(np.transpose(Z, (0,2,1,3)), (data.shape[0]*n_delay, p, n_angle))
        y = np.reshape(combine_complex(np.expand_dims(data[:,p*stride,:], axis=1), n_delay, n_angle), (data.shape[0]*n_delay,n_angle))
    elif mode == "angular_corr" or mode == "multivar_lls":
        # Z = np.reshape(combine_complex(data[:,stack:stack+p*stride:stride,:], n_delay, n_angle), (data.shape[0],p,n_delay,n_angle))
        # y = np.reshape(combine_complex(np.expand_dims(data[:,stack+p*stride,:], axis=1), n_delay, n_angle), (data.shape[0],n_delay,n_angle))
        # (T-stack)-p*stride:(T-stack)+1:stride
        if backwards:
            Z = np.reshape(combine_complex(data[:,e_i-p*stride:e_i:stride,:], n_angle, n_delay, n_chan=n_chan), (data.shape[0],p,n_delay,n_angle))
            y = np.reshape(combine_complex(np.expand_dims(data[:,e_i,:], axis=1), n_angle, n_delay, n_chan=n_chan), (data.shape[0],n_angle,n_delay))
        else:
            Z = np.reshape(combine_complex(data[:,:p*stride:stride,:], n_angle, n_delay, n_chan=n_chan), (data.shape[0],p,n_angle,n_delay))
            y = np.reshape(combine_complex(np.expand_dims(data[:,p*stride,:], axis=1),n_angle, n_delay, n_chan=n_chan), (data.shape[0],n_angle,n_delay))
    if batch_factor != None:
        Z = subsample_batches(Z, batch_factor=batch_factor)
        y = subsample_batches(y, batch_factor=batch_factor)
    return Z, y

def add_batch(data_down, batch, type_str, T, img_channels, img_height, img_width, data_format, n_truncate):
    # concatenate batch data onto end of data
    # Inputs:
    # -> data_up = np.array for uplink
    # -> data_down = np.array for downlink
    # -> batch = mat file to add to np.array
    # -> type_str = part of key to select for training/validation
    x_down = batch['HD_{}'.format(type_str)]    
    x_down = np.reshape(x_down[:,:T,:], get_data_shape(len(x_down), T, img_channels, img_height, img_width, data_format))
    if data_down is None:
        return x_down[:,:,:n_truncate,:] if img_channels > 0 else truncate_flattened_matrix(x_down, img_height, img_width, n_truncate)
    else:
        return np.vstack((data_down,x_down[:,:,:n_truncate,:])) if img_channels > 0 else np.vstack((data_down,truncate_flattened_matrix(x_down, img_height, img_width, n_truncate)))

def split_complex(data,mode=0,T=10):
    if T > 1:
        if mode == 0:
            # default behavior
            re = np.expand_dims(np.real(data).astype('float32'),axis=2) # real portion
            im = np.expand_dims(np.imag(data).astype('float32'),axis=2) # imag portion
            return np.concatenate((re,im),axis=2)
        if mode == 1:
            # written for angular_corr
            re = np.real(data).astype('float32') # real portion
            im = np.imag(data).astype('float32') # imag portion
            return np.concatenate((re,im),axis=1)
    else:
        return np.concatenate((np.expand_dims(np.real(data), axis=1), np.expand_dims(np.imag(data), axis=1)), axis=1)

def truncate_flattened_matrix(x, n_delay, n_angle, n_truncate):
    """
    when img_channels == 0, shape of input is (n_batch, T, 2*n_delay*n_angle)
    need to reshape->truncate->reshape
    """
    n_batch, T, _ = x.shape
    x = np.reshape(x, (n_batch, T, 2, n_delay, n_angle)) # reshape
    x = x[:,:,:,:n_truncate,:] # truncate
    x = np.reshape(x, (n_batch, T, 2*n_truncate*n_angle)) # reshape
    return x

def get_data_shape(samples,T,img_channels,img_height,img_width,data_format):
    if img_channels > 0:
        if(data_format=="channels_last"):
            shape = (samples, T, img_height, img_width, img_channels) if T != 1 else (samples, img_height, img_width, img_channels)
        elif(data_format=="channels_first"):
            shape = (samples, T, img_channels, img_height, img_width) if T != 1 else (samples, img_channels, img_height, img_width)
    else:
        # TODO: get rid of magic number
        shape = (samples, T, 2*img_height*img_width)
    return shape

def subsample_time(data,T,T_max=10):
    """ shorten along T axis """
    if (T < T_max):
        data = data[:,0:T,:]
    return data

def subsample_batches(data,batch_factor=10):
    """ shorten along batch axis """
    N_batch = int(data.shape[0] / batch_factor)
    if N_batch > 0:
        slc = [slice(None)] * len(data.shape) 
        slc[0] = slice(0, N_batch)
        data = data[(slc)]
    return data

def batch_str(base,num):
        return base+str(num)+'.mat'

def stack_data(x1, x2):
    # stack two np arrays with identical shape
    return np.vstack(x1, x2)

def combine_complex(data, height=32, width=32, n_chan=0, T=10):
    assert(n_chan in [0,2])
    if n_chan == 0:
        return data[:,:,:height*width] + data[:,:,height*width:]*1j
    elif n_chan == 2:
        if len(data.shape) == 5:
            return data[:,:,0,:,:] + data[:,:,1,:,:]*1j
        elif len(data.shape) == 4:
            return data[:,0,:,:] + data[:,1,:,:]*1j

def dataset_pipeline(batch_num, debug_flag, aux_bool, dataset_spec, M_1, img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10, train_argv = True,  merge_val_test = True, quant_config = None, idx_split=0, n_truncate=32, total_num_files=21):
    """
    Load and split dataset according to arguments
    Assumes batch-wise splits (i.e., concatenating along axis=0)
    Returns: [data_train, data_val, data_test]
    """
    print(f"aux_bool: {aux_bool}")
    x_train = x_train_up = x_val = x_val_up = x_test = x_test_up = None

    if dataset_spec:
        train_str = dataset_spec[0]
        val_str = dataset_spec[1]
        if len(dataset_spec) ==3:
            test_str = dataset_spec[2]
    else:
        train_str = 'data/data_001/Data100_Htrainin_down_FDD_32ant'
        val_str = 'data/data_001/Data100_Hvalin_down_FDD_32ant'

    # start from split idx*batch_num 
    idx_start = idx_split*batch_num+1 # idx_split is 0-indexed
    if (idx_start > total_num_files) or (idx_start+batch_num > total_num_files): 
        print("=== idx_split too large for given batch_num and total_num_files ===")
        return None

    for batch in range(idx_start, batch_num+idx_start):
        print("--- Adding batch #{} ---".format(batch))
        # mat = sio.loadmat('data/data_001/Data100_Htrainin_down_FDD_32ant_{}.mat'.format(batch))
        if train_argv:
            mat = sio.loadmat(batch_str(train_str,batch))
            x_train  = add_batch(x_train, mat, 'train', T, img_channels, img_height, img_width, data_format, n_truncate)
        mat = sio.loadmat(batch_str(val_str,batch))
        x_val  = add_batch(x_val, mat, 'val', T, img_channels, img_height, img_width, data_format, n_truncate)
        if len(dataset_spec) == 3:
            mat = sio.loadmat(batch_str(test_str,batch))
            x_test  = add_batch(x_test, mat, 'test', T, img_channels, img_height, img_width, data_format, n_truncate)

    if len(dataset_spec) < 3:
        x_test = x_val
        x_test_up = x_val_up

    # bundle training data calls so they are skippable
    if train_argv:
        # x_train = subsample_time(x_train,T)
        x_train = x_train.astype('float32')
        if img_channels > 0:
            x_train = np.reshape(x_train, get_data_shape(len(x_train), T, img_channels, img_height, n_truncate, data_format))  # adapt this if using `channels_first` image data format
        if aux_bool:
            aux_train = np.zeros((len(x_train),M_1))

    # x_val = subsample_time(x_val,T)
    # x_test = subsample_time(x_test,T)

    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    if img_channels > 0:
        x_val = np.reshape(x_val, get_data_shape(len(x_val), T, img_channels, img_height, n_truncate, data_format))  # adapt this if using `channels_first` image data format
        x_test = np.reshape(x_test, get_data_shape(len(x_test), T, img_channels, img_height, n_truncate, data_format))  # adapt this if using `channels_first` image data format

    if aux_bool:
        aux_val = np.zeros((len(x_val),M_1)).astype('float32')
        aux_test = np.zeros((len(x_test),M_1)).astype('float32')

    if (merge_val_test):
        # merge validation and test sets

        if aux_bool:
            aux_val  = np.vstack((aux_val, aux_test))
            aux_test = aux_val
        x_val  = np.vstack((x_val, x_test))
        x_test = x_val

    # concat and (optionally) quantize data
    # TODO: Re-validate. Changed since last run of quantized CSI. 
    quant_bool = type(quant_config) != type(None)
    if quant_bool:
        val_min, val_max, bits = get_keys_from_json(quant_config, keys=['val_min','val_max','bits'])
    if train_argv:
        data_train = x_train if not quant_bool else quantize(x_train,val_min,val_max,bits) 
    data_val = x_val if not quant_bool else quantize(x_val,val_min,val_max,bits) 
    data_test = x_test if not quant_bool else quantize(x_test,val_min,val_max,bits) 
    if aux_bool:
        if train_argv:
            data_train = [aux_train, data_train]
        data_val = [aux_val, data_test]
        data_test = [aux_test, data_test]

    if (not train_argv):
        data_train = None

    # if img_channels > 0:
    #     return data_train[:,:,:n_truncate,:], data_val[:,:,:n_truncate,:], data_test[:,:,:n_truncate,:]
    # else:
    return data_train, data_val, data_test

def dataset_pipeline_col(debug_flag, aux_bool, dataset_spec, M_1, img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10, train_argv = True, quant_config = None, idx_split=0, n_truncate=32, total_num_files=21):
    """
    Load and split dataset according to arguments
    Assumes timeslot splits (i.e., concatenating along axis=1)
    Returns: [data_train, data_val, data_test]
    """
    x_all = x_all_up = None
    assert(len(dataset_spec) == 4)
    dataset_str, dataset_tail, dataset_key, val_split = dataset_spec

    for timeslot in range(1,T+1):
        print("--- Adding batch #{} ---".format(timeslot))
        with h5py.File(f"{dataset_str}{timeslot}_{dataset_tail}", 'r') as f:
            x_t = np.transpose(f[dataset_key][()], [3,2,1,0])
        # x_t = sio.loadmat(f"{dataset_str}{timeslot}_{dataset_tail}")[dataset_key]
        x_all = add_batch_col(x_all, x_t, img_channels, img_height, img_width, data_format, n_truncate)

    # split to train/val
    x_train = x_all[:val_split,:,:,:,:]
    x_val = x_all[val_split:,:,:,:,:]

    # bundle training data calls so they are skippable
    if train_argv:
        # x_train = subsample_time(x_train,T)
        x_train = x_train.astype('float32')
        if img_channels > 0:
            x_train = np.reshape(x_train, get_data_shape(len(x_train), T, img_channels, img_height, n_truncate, data_format))  # adapt this if using `channels_first` image data format
        if aux_bool:
            aux_train = np.zeros((len(x_train),M_1))

    x_val = x_val.astype('float32')

    if img_channels > 0:
        x_val = np.reshape(x_val, get_data_shape(len(x_val), T, img_channels, img_height, n_truncate, data_format))  # adapt this if using `channels_first` image data format

    if aux_bool:
        aux_val = np.zeros((len(x_val),M_1)).astype('float32')

    # concat and (optionally) quantize data
    # TODO: Re-validate. Changed since last run of quantized CSI. 
    quant_bool = type(quant_config) != type(None)
    if quant_bool:
        val_min, val_max, bits = get_keys_from_json(quant_config, keys=['val_min','val_max','bits'])
    if train_argv:
        data_train = x_train if not quant_bool else quantize(x_train,val_min,val_max,bits) 

    data_val = x_val if not quant_bool else quantize(x_val,val_min,val_max,bits) 
    if aux_bool:
        if train_argv:
            data_train = [aux_train, data_train]
        data_val = [aux_val, data_val]

    if (not train_argv):
        data_train = None

    # if img_channels > 0:
    #     return data_train[:,:,:n_truncate,:], data_val[:,:,:n_truncate,:], data_test[:,:,:n_truncate,:]
    # else:
    return data_train, data_val

def add_batch_col(dataset, batch, img_channels, img_height, img_width, data_format, n_truncate):
    # concatenate batch data along time axis 
    # Inputs:
    # -> dataset = np.array for downlink
    # -> batch = mat file to add to np.array
    batch = np.expand_dims(batch, axis=1)
    if dataset is None:
        return batch[:,:,:,:n_truncate] if img_channels > 0 else truncate_flattened_matrix(batch, img_height, img_width, n_truncate)
    else:
        return np.concatenate((dataset, batch[:,:,:,:,:n_truncate]), axis=1) if img_channels > 0 else np.concatenate((dataset,truncate_flattened_matrix(batch, img_height, img_width, n_truncate)), axis=1)

def load_pow_diff(diff_spec,T=1):
    # TODO: load data for T > 1
    pow_diff_val = sio.loadmat(f"{diff_spec[0]}1.mat")["Pow_val"]
    pow_diff_test = sio.loadmat(f"{diff_spec[1]}1.mat")["Pow_test"]
    pow_diff = np.squeeze(np.vstack((pow_diff_val, pow_diff_test)))
    return pow_diff