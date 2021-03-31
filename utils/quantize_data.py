from numpy import digitize, linspace
import numpy as np
import csv

def quantize(target,val_min=-1.0,val_max=1.0,bits=10):
    """
        Quantize a numpy array between [val_min, val_max] with 2**bits levels
    """
    bins = linspace(val_min,val_max,2**bits-1)
    ind = digitize(target, bins)
    return bins[ind-1]

def quantize_int(target,norm="norm_H4",bits=10):
    """
        Quantize a numpy array to integer values from 0 to (2**bits)-1
    """
    if norm == "norm_H4":
        out = ((target + 1) / 2) * 2**bits
    elif norm == "norm_H3":
        out = target * 2**bits
    bins = np.arange(2**bits) 
    out = np.digitize(out, bins)
    return out

def mu_law_companding(x, extrema, dynamic_range_i=32, mu = 255):
    code_abs_max = np.amax(np.absolute(extrema))
    x = x / code_abs_max # normalize by extrema
    x = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu) # apply mu-law
    x = np.round(x*(dynamic_range_i-1)) # quantize using dynamic range
    x = x / (dynamic_range_i-1) # normalize to range [0,1]
    x = np.sign(x) * (1 / mu) *(np.exp(np.abs(x)*np.log(1 + mu)) - 1) # inverse mu-law
    x = x * code_abs_max # scale back to original range
    return x

### helper function: get extrema from dataset splits
def get_minmax(data):
    """
        Get minmax vals from list of data 
    """
    for i in range(len(data)):
        if data[i] != None:
            d_min = min(np.min(data[i]), d_min) if i > 0 else np.min(data[i])
            d_max = max(np.max(data[i]), d_max) if i > 0 else np.max(data[i])
    return [d_min, d_max]

if __name__ == "__main__":
    bits = 8
    print(" --- Testing quantize ---")
    val_min = 0
    val_max = 255
    test = linspace(val_min,val_max,2**bits)
    print("len(test): {}".format(len(test)))
    print("test:")
    print(test)
    out = quantize(test,val_min=val_min,val_max=val_max,bits=bits)
    print("len(out): {}".format(len(out)))
    print("out:")
    print(out)
    print("quant_error:")
    print(test-out)
