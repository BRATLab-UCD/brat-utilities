# pickle_csi_sample_p2d.py

import h5py
import math
import torch
import argparse
import numpy as np 
import pickle as pkl
from tqdm import tqdm
from utils.data_tools import dataset_pipeline_full_batchwise, DatasetToDevice
from utils.unpack_json import get_keys_from_json
from utils.parsing import str2bool
from P2D.modules import P2D_Diag
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
parser.add_argument("-g", "--gpu_num", type=int, default=0, help="number for torch device (cuda:gpu_num)")
# parser.add_argument("-b", "--n_batch", type=int, default=5, help="number of batches to load from target dir")
parser.add_argument("-bo", "--batch_offset", type=int, default=0, help="batch offset w.r.t. mat file numbering")
parser.add_argument("-a", "--angular_bool", type=str2bool, default=True, help="bool for angular domain (alternatively, spatial)")
parser.add_argument("-sz", "--downsample_size", type=int, default=128, help="size after frequency downsampling")

SNR_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
D_list = [1, 2, 4, 8, 16]
delta_list = [0.0, 1e-2, 1e-1, 5e-1, 1.0]

opt = parser.parse_args()
angular_bool = opt.angular_bool
mod_str = "angular" if angular_bool else "spatial"

json_config = "/home/mdelrosa/git/csinet-lstm/config/csinet_lstm_outdoor_cost2100_full.json"

M_1, data_format, network_name, subnetwork_name, model_dir, norm_range, minmax_file, share_bool, T, dataset_spec, diff_spec, batch_num, lr, batch_size, subsample_prop, thresh_idx_path = get_keys_from_json(json_config, keys=['M_1', 'df', 'network_name', 'subnetwork_name', 'model_dir', 'norm_range', 'minmax_file', 'share_bool', 'T', 'dataset_spec', 'diff_spec', 'batch_num', 'lr', 'batch_size', 'subsample_prop', 'thresh_idx_path'])

i_batch = 17
n_batch = 1 # only look at first 5000 samples for now
aux_bool = False
n_subcarriers = 1024
img_channels = 2
crop_height = 32
crop_width = 32
T = 10
sz = opt.downsample_size # 128/1024 = 1/8
device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')

# snippets from dataset_pipeline_full_batchwise
assert(len(dataset_spec) == 5)
dataset_str, dataset_tail, dataset_key, dataset_full_key, val_split = dataset_spec

s_batch = 17
e_batch = s_batch + n_batch

for i_batch in range(s_batch, e_batch):
    i_batch_corrected = i_batch - s_batch
    batch_str = f"{dataset_str}{i_batch}{dataset_tail}"
    if i_batch_corrected >= opt.batch_offset:
        if opt.debug_flag == 0:
            print(f"--- Adding batch #{i_batch} from {batch_str} ---")
            with h5py.File(batch_str, 'r') as f:
                x_t = np.transpose(f[dataset_key][()], [3,2,1,0])
                x_t_full = np.transpose(f[dataset_full_key][()], [3,2,1,0]) 
                f.close()
        elif opt.debug_flag == 1:
            var = 0.0007
            n_all = 500
            shape_all = (n_all,T,crop_height,crop_width)
            shape_all_full = (n_all,T,n_subcarriers,crop_width)
            x_t = np.random.normal(0,var,size=shape_all)+1j*np.random.normal(0,var,size=shape_all)
            x_t_full = np.random.normal(0,var,size=shape_all_full)+1j*np.random.normal(0,var,size=shape_all_full)

        print(f"x_t.shape: {x_t.shape} - x_t_full.shape: {x_t_full.shape}")
        n_all, T, n_subcarriers, n_spatial = x_t_full.shape

        # make p2d sample from x_t_full
        print(f"--- Perform P2D freq with downsample size = {crop_height}x{sz} - CR={sz/n_subcarriers} ---")

        x_t = x_t.view("complex")
        x_t_full = x_t_full.view("complex")
        x_t = torch.complex(torch.Tensor(np.real(x_t)), torch.Tensor(np.imag(x_t)))
        x_t_full = torch.complex(torch.Tensor(np.real(x_t_full)), torch.Tensor(np.imag(x_t_full)))

        data_all_trunc = torch.zeros((n_all, T, crop_height, crop_width), dtype=x_t.dtype) # to torch
        freq_all = torch.zeros_like(x_t_full)
        pow_diff = torch.zeros(n_all, T)

        data_loader = DataLoader(dataset=DatasetToDevice(x_t_full, n_all, device), batch_size=batch_size, num_workers=0)

        print(f"--- Downsampling data for T timeslots with DR_f = {sz}/{n_subcarriers} ---")
        for i, (data_real, data_imag) in tqdm(enumerate(data_loader)):
            idx_s = i*batch_size
            idx_e = min(idx_s+batch_size, n_all)
            idx_sz = idx_e - idx_s
            data_idx = torch.complex(data_real, data_imag)
            if angular_bool:
                data_idx = torch.fft.fft(data_idx, dim=3) # spatial->angular
            freq_all[idx_s:idx_e,:] = torch.fft.fft(data_idx, dim=2) # delay->freq
            data_idx_trunc = data_idx[:,:,:crop_height,:]
            data_all_trunc[idx_s:idx_e,:] = data_idx_trunc

            # calculate pow_diff in angular (or spatial) domain
            data_full_sq = torch.real(data_idx*torch.conj(data_idx)).view((idx_sz, T, -1))
            data_trunc_sq = torch.real(data_idx_trunc*torch.conj(data_idx_trunc)).view((idx_sz, T, -1))
            pow_full = torch.sum(data_full_sq, dim=2)
            pow_trunc = torch.sum(data_trunc_sq, dim=2)
            pow_diff_idx = pow_full - pow_trunc
            pow_diff[idx_s:idx_e,:] = pow_diff_idx
            assert(torch.sum(torch.where(pow_diff_idx > 0, 1, 0), (0,1)) == idx_sz*T)
    
        del data_loader, x_t_full, data_full_sq, data_trunc_sq, pow_diff_idx, pow_trunc
        torch.cuda.empty_cache()

        freq_all = torch.transpose(freq_all, 3, 2)
        data_all_trunc = torch.transpose(data_all_trunc, 3, 2)
        data_all_trunc = data_all_trunc.numpy()
        freq_all = freq_all.numpy()
        pow_diff = pow_diff.numpy()

        for D in D_list:
            for delta in delta_list:
                model = P2D_Diag(sz, n_subcarriers, crop_height, D)
                model.fit(delta=delta) # fit model with regularization 
        
                freq_all_t = freq_all[:,0,:]
                if D == 1:
                    freq_all_down = np.dot(freq_all_t, model.P[0,:].T)
                else:
                    freq_all_down = np.zeros((n_all, n_spatial, sz), dtype="complex")
                    tqdm_desc = f"t_{1}: Downsampling Size to {n_spatial}x{sz} using {D} diagonal pilots"
                    for i in tqdm(range(n_all), desc=tqdm_desc):
                        freq_all_down[i,:,:] = model.downsample(freq_all_t[i,:,:])

                shape_freq_down = freq_all_down.shape

                print(f"--- Sweep through SNR values ({SNR_list}) with DR_f = {sz}/{n_subcarriers}, D={D} ---")

                freq_all_down_pow = np.real(np.sum(np.sum(np.conj(freq_all_down)*freq_all_down, axis=2), axis=1))
                y_test = data_all_trunc


                MSE = [0]*len(SNR_list)
                NMSE = [0]*len(SNR_list)

                for j, snr in enumerate(SNR_list):
                    MSE_batch = 0
                    NMSE_batch = 0
                    noise_down = np.random.normal(0,1.0,size=shape_freq_down)+1j*np.random.normal(0,1.0,size=shape_freq_down)
                    noise_pow = np.sum(np.sum(np.conj(noise_down)*noise_down, axis=2), axis=1)
                    current_snr = freq_all_down_pow / noise_pow
                    snr_scale = current_snr / (10**(snr/10))
                    noise_down = np.reshape(np.reshape(noise_down, (n_all, -1)) * snr_scale[:, None], shape_freq_down)
                    freq_all_down_noisy = freq_all_down + noise_down
                    # y_hat_t = model.predict(freq_all_down_noisy)
                    # y_err = y_test[i,0,:] - y_hat_t

                    for i in tqdm(range(n_all), desc=f"-> initial accuracy for DR_f={sz}/{n_subcarriers}, Diag={D}, SNR={snr}"):
                        y_hat_t = model.predict(freq_all_down_noisy[i,:])
                        y_err = y_test[i,0,:] - y_hat_t
                        SE = np.real(np.sum(np.real(y_err)**2) + np.sum(np.imag(y_err)**2))
                        MSE_batch += SE / n_all
                        NMSE_batch += SE / (freq_all_down_pow[i]*n_all)
                        # y_hat[i,t,:,:] = y_hat_t

                    MSE[j] = MSE_batch 
                    NMSE[j] = NMSE_batch 
        
                print(f"--- D={D} | delta={delta} | mse/nmse results for SNR levels ---")
                for i, (snr, mse_snr, nmse_snr) in enumerate(zip(SNR_list, MSE, NMSE)):
                    print(f"-> SNR={snr}dB |  MSE (denormalized {mod_str} domain): {mse_snr:4.3E}")
                    print(f"-> SNR={snr}dB | NMSE (denormalized {mod_str} domain): {10*np.log10(nmse_snr):4.3f}dB")

                # pickle results 
                pkl_dict = {
                    "SNR": SNR_list,
                    "MSE": MSE,
                    "NMSE": NMSE
                }

                file_loc = f"results/{mod_str}_ibatch{i_batch}_samples{n_all}_D{D}_sz{sz}_delta{delta:3.2f}.pkl"
                with open(file_loc, "wb") as f:
                    pkl.dump(pkl_dict, f)
                    f.close()

    else:
        print(f"--- skipping {batch_str} ---")