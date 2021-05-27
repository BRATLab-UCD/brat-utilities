import argparse
import pickle
import copy
import sys
import numpy as np

sys.path.append(".")
from utils.NMSE_performance import renorm_H4, renorm_sphH4
from utils.data_tools import dataset_pipeline_col, subsample_batches
from utils.parsing import str2bool
from utils.timing import Timer
from utils.unpack_json import get_keys_from_json
from utils.trainer import fit, score, save_predictions, save_checkpoint_history

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
parser.add_argument("-b", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
parser.add_argument("-tr", "--train_argv", type=str2bool, default=True, help="flag for toggling training")
parser.add_argument("-sp", "--split", type=int, default=0, help="split of entire dataset. must be less than int(<total_num_files> / <n_batch>).")
parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
parser.add_argument("-dt", "--data_type", type=str, default="norm_H4", help="type of dataset to train on (norm_H4, norm_sphH4)")
opt = parser.parse_args()

# dataset pipeline vars 
json_config = "config/csinet_indoor_cost2100_pow.json" if opt.env == "indoor" else "config/csinet_outdoor_cost2100_pow.json"

# unpack config file
dataset_spec, img_channels, data_format, T, n_delay, diff_spec = get_keys_from_json(json_config, keys=["dataset_spec", "img_channels", "data_format", "T", "n_delay", "diff_spec"])
input_dim = (2,n_delay,32)
aux_bool = False
aux_size = 0

# load data
pow_diff, data_train, data_val = dataset_pipeline_col(opt.debug_flag, aux_bool, dataset_spec, diff_spec, aux_size, T = T, img_channels = input_dim[0], img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, train_argv = opt.train_argv)

print(f"--- data_train.shape = {data_train.shape} with range {np.min(data_train)} to {np.max(data_train)} ---")
print(f"--- data_val.shape = {data_val.shape} with range {np.min(data_val)} to {np.max(data_val)} ---")