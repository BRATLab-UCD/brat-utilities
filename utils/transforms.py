from torchvision import transforms
import numpy as np
import torch

class CircularShift(object):
    """Rescale the image in a sample to a given size.

    Args:
	    shift_axis (int): Axis along which to perform circular shift.
          Batch dim is dropped in Dataset object, so this needs to be
          adjusted by -1.
	    proba (float): Probability of random circular shift
    """

    def __init__(self, shift_axis, proba):
        # assert isinstance(output_size, (int, tuple))
        assert isinstance(shift_axis, int)
        assert (isinstance(proba, float) and (proba >= 0.0) and (proba <= 1.0))
        # self.output_size = output_size
        self.shift_axis = shift_axis - 1  # axis *ignores* batch dim -> subtract 1
        self.proba = proba

    def __call__(self, sample):
        x = np.random.uniform(0,1)
        if (x < self.proba):
            # get max val for roll
            max_shift = sample.size(self.shift_axis)
            shift_amount = np.random.randint(0,max_shift)
            return torch.roll(sample, shift_amount, self.shift_axis)
        else:
            return sample