import scipy.io as sio
import numpy as np
import h5py
# import os # for checking process memory
import pickle

def get_t1_power_col(dataset_spec, outpath, stride=1, T=10, img_channels=2, img_height=32, img_width=32, mat_type=0):
    # iterate through timeslots, store:
    # 1. running min and max of each timeslot
    # 2. power for each timeslot
    assert(len(dataset_spec) == 4)
    dataset_str, dataset_tail, dataset_key, val_split = dataset_spec

    H_down_min_pre = np.zeros(T)
    H_down_max_pre = np.zeros(T)
    H_down_min = np.zeros(T)
    H_down_max = np.zeros(T)
    for timeslot in range(1,T*stride+1,stride):
        batch_str = f"{dataset_str}{timeslot}_{dataset_tail}"
        print(f"--- Adding batch #{timeslot} from {batch_str} ---")
        if mat_type == 0:
            with h5py.File(batch_str, 'r') as f:
                x_t = np.transpose(f[dataset_key][()], [3,2,1,0])
                f.close()
        elif mat_type == 1:
            mat = sio.loadmat(batch_str)
            x_t = mat[dataset_key]
            x_t = np.reshape(x_t, (x_t.shape[0], img_channels, img_height, img_width))
        else:
            print("--- Unrecognized mat_type ---")
            return None
        if timeslot == 1:
            samples, num_channels, img_height, img_width = x_t.shape

        # first, get power of first timeslot
        # Hur_up = np.reshape(Hur_up, (samples, T, -1))
        x_t = np.reshape(x_t, (samples, -1))
        pow_down = np.sqrt(np.sum(x_t**2, axis=1))
        # pow_up = np.sqrt(np.sum(H_t_up**2, axis=1))
        if timeslot == 1:
            # print(f"H_t_down.shape: {H_t_down.shape} - H_t_up.shape: {H_t_up.shape} - pow_down.shape: {pow_down.shape} - pow_up.shape: {pow_up.shape}")
            # pow_t1_up = pow_up
            pow_t1_down = pow_down
            print(f"pow_t1_down range: {np.min(pow_t1_down)} to {np.max(pow_t1_down)} -- pow_t1_down.shape: {pow_t1_down.shape}")
        # pickle_dict = {"pow_up": pow_up, "pow_down": pow_down}
        pickle_dict = {"pow_down": pow_down}
        with open(f"{outpath}/H_t{timeslot}_power.pkl", "wb") as f:
            pickle.dump(pickle_dict, f)
            f.close()

        # second, iterate over each timeslot, normalize by pow_t1, and calculate extrema per timeslot

        j = timeslot - 1 # alias for timeslot - 1
        H_down_min_pre[j] = np.min(x_t)
        H_down_max_pre[j] = np.max(x_t)

        # somehow, this might be the problem?
        norm_down = x_t / pow_t1_down[:,None]

        H_down_min[j] = np.min(norm_down)
        H_down_max[j] = np.max(norm_down)
        print(f"H_down_min: {H_down_min[j]} - H_down_max: {H_down_max[j]} - H_down_min: {H_down_min_pre[j]} - H_down_max: {H_down_max_pre[j]}")

        
    for i in range(T):
        print(f"t{i+1}: sph_min={H_down_min[i]} - sph_max={H_down_max[i]} - pre_min={H_down_min_pre[i]} - pre_max={H_down_max_pre[i]}")

    # extrema_dict = {"H_down_ext": [H_down_min, H_down_max], "H_up_ext": [H_up_min, H_up_max],}
    extrema_dict = {"H_down_ext": [H_down_min, H_down_max]}
    with open(f"{outpath}/H_timeslot_extrema_sph.pkl", "wb") as f:
        pickle.dump(extrema_dict, f)
        f.close()

    # extrema_dict_pre = {"H_down_ext": [H_down_min_pre, H_down_max_pre], "H_up_ext": [H_up_min_pre, H_up_max_pre],}
    extrema_dict_pre = {"H_down_ext": [H_down_min_pre, H_down_max_pre]}
    with open(f"{outpath}/H_timeslot_extrema_pre.pkl", "wb") as f:
        pickle.dump(extrema_dict_pre, f)
        f.close()

def get_t1_power_col_mag(dataset_spec, outpath, stride=1, T=10, mat_type=0, img_channels=2, img_height=32, img_width=32):
    # magnitude-based spherical normalization -- get power 
    # iterate through timeslots, store:
    # 1. running min and max of each timeslot
    # 2. power for each timeslot
    assert(len(dataset_spec) == 5)
    dataset_str, dataset_tail, key_down, key_up, val_split = dataset_spec

    H_down_min_pre = np.zeros(T)
    H_down_max_pre = np.zeros(T)
    H_down_min = np.zeros(T)
    H_down_max = np.zeros(T)
    H_mag_down_min = np.zeros(T)
    H_mag_down_max = np.zeros(T)
    H_up_min_pre = np.zeros(T)
    H_up_max_pre = np.zeros(T)
    H_up_min = np.zeros(T)
    H_up_max = np.zeros(T)
    H_mag_up_min = np.zeros(T)
    H_mag_up_max = np.zeros(T)
    for timeslot in range(1,T*stride+1,stride):
        batch_str = f"{dataset_str}{timeslot}_{dataset_tail}"
        print(f"--- Adding batch #{timeslot} from {batch_str} ---")
        if mat_type == 0:
            with h5py.File(batch_str, 'r') as f:
                x_t = np.transpose(f[key_down][()], [3,2,1,0])
                x_t_u = np.transpose(f[key_up][()], [3,2,1,0])
                f.close()
        elif mat_type == 1:
            mat = sio.loadmat(batch_str)
            x_t = mat[key_down]
            x_t = np.reshape(x_t, (x_t.shape[0], img_channels, img_height, img_width))
            x_t_u = mat[key_up]
            x_t_u = np.reshape(x_t_u, (x_t_u.shape[0], img_channels, img_height, img_width))
        else:
            print("--- Unrecognized mat_type ---")
            return None

        if timeslot == 1:
            samples, num_channels, n_delay, n_angle = x_t.shape

        # first, get power of first timeslot
        # Hur_up = np.reshape(Hur_up, (samples, T, -1))
        x_t = np.reshape(x_t, (samples, -1))
        pow_down = np.sqrt(np.sum(x_t**2, axis=1))
        x_t_u = np.reshape(x_t_u, (samples, -1))
        pow_down = np.sqrt(np.sum(x_t**2, axis=1))
        pow_up = np.sqrt(np.sum(x_t_u**2, axis=1))
        # pow_up = np.sqrt(np.sum(H_t_up**2, axis=1))
        if timeslot == 1:
            # print(f"H_t_down.shape: {H_t_down.shape} - H_t_up.shape: {H_t_up.shape} - pow_down.shape: {pow_down.shape} - pow_up.shape: {pow_up.shape}")
            # pow_t1_up = pow_up
            pow_t1_down, pow_t1_up = pow_down, pow_up
            print(f"pow_t1_down range: {np.min(pow_t1_down)} to {np.max(pow_t1_down)} -- pow_t1_down.shape: {pow_t1_down.shape}")
            print(f"pow_t1_up range: {np.min(pow_t1_up)} to {np.max(pow_t1_up)} -- pow_t1_up.shape: {pow_t1_up.shape}")
        # pickle_dict = {"pow_up": pow_up, "pow_down": pow_down}
        pickle_dict = {"pow_down": pow_down, "pow_up": pow_up}
        with open(f"{outpath}/H_t{timeslot}_power.pkl", "wb") as f:
            pickle.dump(pickle_dict, f)
            f.close()

        # second, iterate over each timeslot, normalize by pow_t1, and calculate extrema per timeslot

        j = timeslot - 1 # alias for timeslot - 1
        H_down_min_pre[j] = np.min(x_t)
        H_down_max_pre[j] = np.max(x_t)
        H_up_min_pre[j] = np.min(x_t_u)
        H_up_max_pre[j] = np.max(x_t_u)

        # normalize by power
        norm_down = x_t / pow_t1_down[:,None]
        norm_up = x_t_u / pow_t1_up[:,None]

        H_down_min[j] = np.min(norm_down)
        H_down_max[j] = np.max(norm_down)
        H_up_min[j] = np.min(norm_up)
        H_up_max[j] = np.max(norm_up)

        norm_down = np.reshape(norm_down, (samples, num_channels, n_delay, n_angle))
        norm_up = np.reshape(norm_up, (samples, num_channels, n_delay, n_angle))
        # H_mag_down = np.sqrt(norm_down[:,0,:,:]**2, norm_down[:,1,:,:]**2)
        # H_mag_up   = np.sqrt(norm_up[:,0,:,:]**2, norm_up[:,1,:,:]**2)
        H_mag_down = np.absolute(norm_down[:,0,:,:]+norm_down[:,1,:,:]*1j)
        H_mag_up = np.absolute(norm_up[:,0,:,:]+norm_up[:,1,:,:]*1j)
        H_mag_down_min[j] = np.min(H_mag_down)
        H_mag_down_max[j] = np.max(H_mag_down)
        H_mag_up_min[j] = np.min(H_mag_up) 
        H_mag_up_max[j] = np.max(H_mag_up)
        print(f"H_down_min: {H_down_min[j]} - H_down_max: {H_down_max[j]} - H_down_min: {H_down_min_pre[j]} - H_down_max: {H_down_max_pre[j]} - H_mag_down_min: {H_mag_down_min[j]} - H_mag_down_max: {H_mag_down_max[j]}")
        print(f"H_up_min: {H_up_min[j]} - H_up_max: {H_up_max[j]} - H_up_min: {H_up_min_pre[j]} - H_up_max: {H_up_max_pre[j]}- H_mag_up_min: {H_mag_up_min[j]} - H_mag_up_max: {H_mag_up_max[j]}")

    for i in range(T):
        print(f"t{i+1} down: sph_min={H_down_min[i]} - sph_max={H_down_max[i]} - pre_min={H_down_min_pre[i]} - pre_max={H_down_max_pre[i]} - mag_min={H_mag_down_min} - mag_max={H_mag_down_max}")
        print(f"t{i+1}   up: sph_min={H_up_min[i]} - sph_max={H_up_max[i]} - pre_min={H_up_min_pre[i]} - pre_max={H_up_max_pre[i]} - mag_min={H_mag_up_min} - mag_max={H_mag_up_max}")

    # extrema_dict = {"H_down_ext": [H_down_min, H_down_max], "H_up_ext": [H_up_min, H_up_max],}
    extrema_dict = {"H_down_ext": [H_down_min, H_down_max], "H_up_ext": [H_up_min, H_up_max]}
    with open(f"{outpath}/H_timeslot_extrema_sph.pkl", "wb") as f:
        pickle.dump(extrema_dict, f)
        f.close()

    # extrema_dict = {"H_down_ext": [H_down_min, H_down_max], "H_up_ext": [H_up_min, H_up_max],}
    extrema_dict = {"H_down_ext": [H_mag_down_min, H_mag_down_max], "H_up_ext": [H_mag_up_min, H_mag_up_max]}
    with open(f"{outpath}/H_timeslot_extrema_sph_mag.pkl", "wb") as f:
        pickle.dump(extrema_dict, f)
        f.close()

    # extrema_dict_pre = {"H_down_ext": [H_down_min_pre, H_down_max_pre], "H_up_ext": [H_up_min_pre, H_up_max_pre],}
    extrema_dict_pre = {"H_down_ext": [H_down_min_pre, H_down_max_pre], "H_up_ext": [H_up_min_pre, H_up_max_pre]}
    with open(f"{outpath}/H_timeslot_extrema_pre.pkl", "wb") as f:
        pickle.dump(extrema_dict_pre, f)
        f.close()

def get_t1_col_mag(dataset_spec, outpath, stride=1, T=10, mat_type=0, img_channels=2, img_height=32, img_width=32):
    # magnitude-based minmax normalization 
    # iterate through timeslots, store:
    # 1. running min and max of each timeslot
    assert(len(dataset_spec) == 5)
    dataset_str, dataset_tail, key_down, key_up, val_split = dataset_spec

    H_down_min = np.zeros(T)
    H_down_max = np.zeros(T)
    H_mag_down_min = np.zeros(T)
    H_mag_down_max = np.zeros(T)
    H_up_min = np.zeros(T)
    H_up_max = np.zeros(T)
    H_mag_up_min = np.zeros(T)
    H_mag_up_max = np.zeros(T)
    for timeslot in range(1,T*stride+1,stride):
        batch_str = f"{dataset_str}{timeslot}_{dataset_tail}"
        print(f"--- Adding batch #{timeslot} from {batch_str} ---")
        if mat_type == 0:
            with h5py.File(batch_str, 'r') as f:
                x_t = np.transpose(f[key_down][()], [3,2,1,0])
                x_t_u = np.transpose(f[key_up][()], [3,2,1,0])
                f.close()
        elif mat_type == 1:
            mat = sio.loadmat(batch_str)
            x_t = mat[key_down]
            x_t = np.reshape(x_t, (x_t.shape[0], img_channels, img_height, img_width))
            x_t_u = mat[key_up]
            x_t_u = np.reshape(x_t_u, (x_t_u.shape[0], img_channels, img_height, img_width))
        else:
            print("--- Unrecognized mat_type ---")
            return None

        if timeslot == 1:
            samples, num_channels, n_delay, n_angle = x_t.shape

        # iterate over each timeslot, and calculate magnitude extrema per timeslot

        j = timeslot - 1 # alias for timeslot - 1
        H_down_min[j] = np.min(x_t)
        H_down_max[j] = np.max(x_t)
        H_up_min[j] = np.min(x_t_u)
        H_up_max[j] = np.max(x_t_u)

        # H_mag_down = np.sqrt(norm_down[:,0,:,:]**2, norm_down[:,1,:,:]**2)
        # H_mag_up   = np.sqrt(norm_up[:,0,:,:]**2, norm_up[:,1,:,:]**2)
        H_mag_down = np.absolute(x_t[:,0,:,:]+x_t[:,1,:,:]*1j)
        H_mag_up = np.absolute(x_t_u[:,0,:,:]+x_t_u[:,1,:,:]*1j)
        H_mag_down_min[j] = np.min(H_mag_down)
        H_mag_down_max[j] = np.max(H_mag_down)
        H_mag_up_min[j] = np.min(H_mag_up) 
        H_mag_up_max[j] = np.max(H_mag_up)
        print(f"H_down_min: {H_down_min[j]} - H_down_max: {H_down_max[j]} - H_mag_down_min: {H_mag_down_min[j]} - H_mag_down_max: {H_mag_down_max[j]}")
        print(f"H_up_min: {H_up_min[j]} - H_up_max: {H_up_max[j]} - H_mag_up_min: {H_mag_up_min[j]} - H_mag_up_max: {H_mag_up_max[j]}")

    for i in range(T):
        print(f"t{i+1} down: pre_min: {H_down_min[j]} - pre_max: {H_down_max[j]} - mag_min={H_mag_down_min} - mag_max={H_mag_down_max}")
        print(f"t{i+1}   up: pre_min: {H_up_min[j]} - pre_max: {H_up_max[j]} - mag_min={H_mag_up_min} - mag_max={H_mag_up_max}")

    extrema_dict = {"H_down_ext": [H_mag_down_min, H_mag_down_max], "H_up_ext": [H_mag_up_min, H_mag_up_max]}
    with open(f"{outpath}/H_timeslot_extrema_mag.pkl", "wb") as f:
        pickle.dump(extrema_dict, f)
        f.close()


def get_t1_pow_Kusers(dataset_spec, outpath, n_batch=10, K=2, T=10, mat_type=0, img_channels=2, img_height=32, img_width=32, batches=None):
    # get extrema for minmax/spherical normalization 
    # iterate through timeslots, store:
    # 1. running min and max of each timeslot
    assert(len(dataset_spec) == 5)
    dataset_str, dataset_tail, key_down, key_up, val_split = dataset_spec

    H_down_min = np.zeros((K,T))
    H_down_max = np.zeros((K,T))
    H_sph_down_min = np.zeros((K,T))
    H_sph_down_max = np.zeros((K,T))
    H_up_min = np.zeros((K,T))
    H_up_max = np.zeros((K,T))
    H_sph_up_min = np.zeros((K,T))
    H_sph_up_max = np.zeros((K,T))

    batches = range(1,n_batch+1) if type(batches) is type(None) else batches
    for batch in batches:
        batch_str = f"{dataset_str}{batch}_{dataset_tail}"
        if mat_type == 0:
            print(f"--- Adding batch #{batch} from {batch_str} ---")
            with h5py.File(batch_str, 'r') as f:
                x_t = np.transpose(f[key_down][()], [3,2,1,0])
                x_t_u = np.transpose(f[key_up][()], [3,2,1,0])
                f.close()
        elif mat_type == 1:
            mat = sio.loadmat(batch_str)
            x_t = mat[key_down]
            x_t = np.reshape(x_t, (x_t.shape[0], img_channels, img_height, img_width))
            x_t_u = mat[key_up]
            x_t_u = np.reshape(x_t_u, (x_t_u.shape[0], img_channels, img_height, img_width))
        elif mat_type == 2:
            down_str = f"{dataset_str}{batch}_down{dataset_tail}"
            print(f"--- Adding batch #{batch} from {down_str} ---")
            mat = sio.loadmat(down_str)
            x_t = mat[key_down]
            x_t = np.reshape(x_t, (x_t.shape[0], K, T, img_height, img_width))
            x_t = np.fft.ifft(x_t, axis=3)
            up_str = f"{dataset_str}{batch}_up{dataset_tail}"
            print(f"--- Adding batch #{batch} from {up_str} ---")
            mat = sio.loadmat(up_str)
            x_t_u = mat[key_up]
            x_t_u = np.reshape(x_t_u, (x_t.shape[0], K, T, img_height, img_width))
            x_t_u = np.fft.ifft(x_t_u, axis=3)
        else:
            print("--- Unrecognized mat_type ---")
            return None

        if batch == 1:
            samples, K, T, n_delay, n_angle = x_t.shape
            all_down_pow = np.zeros((samples*n_batch,K))
            all_up_pow = np.zeros((samples*n_batch,K))
        x_t = np.concatenate([np.expand_dims(np.real(x_t), axis=3), np.expand_dims(np.imag(x_t), axis=3)], axis=3)
        x_t_u = np.concatenate([np.expand_dims(np.real(x_t_u), axis=3), np.expand_dims(np.imag(x_t_u), axis=3)], axis=3)

        # iterate over each timeslot, and calculate extrema per timeslot

        for t in range(T):
            # minmax
            x_down_t = x_t[:,:,t,:,:,:]
            x_up_t = x_t_u[:,:,t,:,:,:]

            # sph
            x_down_flat = np.reshape(x_down_t, (x_down_t.shape[:2])+(-1,))
            x_up_flat = np.reshape(x_up_t, (x_up_t.shape[:2])+(-1,))
            if t == 0:
                x_down_pow = np.sqrt(np.sum(x_down_flat*x_down_flat, axis=2))
                x_up_pow = np.sqrt(np.sum(x_up_flat*x_up_flat, axis=2))
                all_down_pow[(batch-1)*samples:batch*samples,:] = x_down_pow
                all_up_pow[(batch-1)*samples:batch*samples,:] = x_up_pow
            for k in range(K):
                H_down_min[k,t] = np.min([np.min(x_down_t[:,k,:]), H_down_min[k,t]])
                H_down_max[k,t] = np.max([np.max(x_down_t[:,k,:]), H_down_max[k,t]])
                H_up_min[k,t] = np.min([np.min(x_up_t[:,k,:]), H_up_min[k,t]])
                H_up_max[k,t] = np.max([np.max(x_up_t[:,k,:]), H_up_max[k,t]])
                x_sph_down = x_down_flat[:,k,:] / x_down_pow[:,k,None]
                x_sph_up   = x_up_flat[:,k,:] / x_up_pow[:,k,None]
                H_sph_down_min[k,t] = np.min([np.min(x_sph_down), H_sph_down_min[k,t]])
                H_sph_down_max[k,t] = np.max([np.max(x_sph_down), H_sph_down_max[k,t]])
                H_sph_up_min[k,t] = np.min([np.min(x_sph_up), H_sph_up_min[k,t]])
                H_sph_up_max[k,t] = np.max([np.max(x_sph_up), H_sph_up_max[k,t]])
                print(f"k={k} - t={t} - H_down_min: {H_down_min[k,t]} - H_down_max: {H_down_max[k,t]} - H_sph_down_min: {H_sph_down_min[k,t]} - H_sph_down_max: {H_sph_down_max[k,t]}")
                print(f"k={k} - t={t} - H_up_min: {H_up_min[k,t]} - H_up_max: {H_up_max[k,t]} - H_sph_up_min: {H_sph_up_min[k,t]} - H_sph_up_max: {H_sph_up_max[k,t]}")

    for i in range(T):
        for k in range(K):
            print(f"t{i+1} k={k+1} - down: pre_min: {H_down_min[k,i]} - pre_max: {H_down_max[k,i]} - sph_min={H_sph_down_min[k,i]} - sph_max={H_sph_down_max[k,i]}")
            print(f"t{i+1} k={k+1} - up: pre_min: {H_up_min[k,i]} - pre_max: {H_up_max[k,i]} - sph_min={H_sph_up_min[k,i]} - sph_max={H_sph_up_max[k,i]}")

    pickle_dict = {"pow_down": all_down_pow, "pow_up": all_up_pow}
    with open(f"{outpath}/H_t1_power.pkl", "wb") as f:
        pickle.dump(pickle_dict, f)
        f.close()

    # spherical norm dict
    extrema_dict = {"H_down_ext": [H_sph_down_min, H_sph_down_max], "H_up_ext": [H_sph_up_min, H_sph_up_max]}
    with open(f"{outpath}/H_timeslot_extrema_sph.pkl", "wb") as f:
        pickle.dump(extrema_dict, f)
        f.close()

    # minmax norm dict
    extrema_dict_pre = {"H_down_ext": [H_down_min, H_down_max], "H_up_ext": [H_up_min, H_up_max]}
    with open(f"{outpath}/H_timeslot_extrema_pre.pkl", "wb") as f:
        pickle.dump(extrema_dict_pre, f)
        f.close()
