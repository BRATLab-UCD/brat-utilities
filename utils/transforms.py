from torchvision import transforms
import numpy as np
import torch

class CircularShift(object):
	"""
	randomly circular shift along given axis with probability

	Args:
		shift_axis (int): Axis along which to perform circular shift.
		  Batch dim is dropped in Dataset object, so this needs to be
		  adjusted by -1.
		proba (float): Probability of random circular shift
	"""

	def __init__(self, shift_axis, proba, dtype=torch.float):
		# assert isinstance(output_size, (int, tuple))
		assert isinstance(shift_axis, int)
		assert (isinstance(proba, float) and (proba >= 0.0) and (proba <= 1.0))
		self.dtype = dtype
		# self.output_size = output_size
		self.shift_axis = shift_axis - 1  # axis *ignores* batch dim -> subtract 1
		self.proba = proba

	def __call__(self, sample):
		x = np.random.uniform(0,1)
		if (x < self.proba):
			# get max val for roll
			max_shift = sample.size(self.shift_axis)
			shift_amount = np.random.randint(0,max_shift)
			out = torch.roll(sample, shift_amount, self.shift_axis)
			if self.dtype is torch.float:
				out = out.float()
			return out
		else:
			if self.dtype is torch.float:
				out = sample.float()
			return out

class ZeroAxis(object):
	"""Randomly set rows/cols of tensor to all-zero

	Args:
		shift_axis (int): Axis along which to perform circular shift.
		  Batch dim is dropped in Dataset object, so this needs to be
		  adjusted by -1.
		proba (float): Probability of random augmentation
		prop (float): Proportion of rows to set to zero
	"""

	def __init__(self, zero_axis, proba, prop, with_mask=False, device="cpu"):
		assert isinstance(zero_axis, int)
		assert (isinstance(proba, float) and (proba >= 0.0) and (proba <= 1.0))
		assert (isinstance(prop, float) and (prop >= 0.0) and (prop <= 1.0))
		assert (isinstance(with_mask, bool))
		self.zero_axis = zero_axis - 1  # axis *ignores* batch dim -> subtract 1
		self.proba = proba
		self.prop = prop
		self.device = device
		self.with_mask = with_mask
		self.mask = None
		self.mask_unsqueeze = None

	def __call__(self, sample):
		x = np.random.uniform(0,1)
		if (x < self.proba):
			c, h, w = sample.size()
			zero_sz = sample.size(self.zero_axis)
			if self.mask is None:
				self.mask = torch.ones(size=(c,zero_sz)).to(self.device)
				self.mask_unsqueeze = torch.ones(size=(c,h,w)).to(self.device)
			self.mask[:] = 1
			zero_num = int(self.prop*zero_sz)
			zero_idx = np.random.choice(zero_sz, size=zero_num, replace=False)
			self.mask[:,zero_idx] = 0
			if self.zero_axis == 1: # height axis
				self.mask_unsqueeze = self.mask.unsqueeze(2).repeat(1,1,w)
			if self.zero_axis == 2: # width axis
				self.mask_unsqueeze = self.mask.unsqueeze(1).repeat(1,h,1)
			zero_sample = self.mask_unsqueeze*sample
			if self.with_mask:
				zero_sample = torch.cat([zero_sample, self.mask_unsqueeze[0,:].unsqueeze(0)])
			return (sample, zero_sample)
		else:
			c, h, w = sample.size()
			if self.mask is None:
				self.mask_unsqueeze = torch.ones(size=(c,h,w)).to(self.device)
			if self.with_mask:
				self.mask_unsqueeze[:] = 1
				zero_sample = torch.cat([sample, self.mask_unsqueeze[0,:].unsqueeze(0)])
				return (sample, zero_sample)
			else:
				return (sample, sample)

class RandomCrop(object):
	"""Randomly crop tensor

	Args:
		crop_axes (int, int): Axis along which to perform circular shift.
		  Batch dim is dropped in Dataset object, so this needs to be
		  adjusted by -1.
		proba (float): Probability of random cropping
		crop_dims (int, int): Size of random cropping to return
	"""

	def __init__(self, crop_dims):
		#TODO: impl explicit crop dims, if necessary
		# assert isinstance(output_size, (int, tuple))
		assert isinstance(crop_dims, tuple)
		# self.output_size = output_size
		self.crop_dims = crop_dims
		self.max_idx = None

	def __call__(self, sample):
		if self.max_idx == None:
			x_lim, y_lim = sample.size(1), sample.size(2)
			self.x_rng = x_lim - self.crop_dims[0]
			self.y_rng = y_lim - self.crop_dims[1]
			assert(self.x_rng >= 0 and self.y_rng >= 0)
		x_s = np.random.randint(self.x_rng) if self.x_rng > 0 else 0
		x_e = x_s + self.crop_dims[0]
		y_s = np.random.randint(self.y_rng) if self.y_rng > 0 else 0
		y_e = y_s + self.crop_dims[1]
		return sample[:,x_s:x_e,y_s:y_e]

class RandomPhase(torch.nn.Module):
	"""
	assuming two-channel real/imag input, fix magnitude and randomize phase of all elements
	phase drawn from uniform random distribution U ~ [0,2\pi]
	"""

	def __init__(self, theta_min=0, theta_max=np.pi, proba=0.5):
		self.theta_min = theta_min
		self.theta_max = theta_max
		self.proba = proba

	def __call__(self, sample):
		foo = np.random.uniform(0,1)
		if foo < self.proba:
			sample_c = sample[0,:,:] + 1j*sample[1,:,:]
			e_rand = torch.FloatTensor(sample_c.size(0), sample.size(1)).uniform_(self.theta_min, self.theta_max)
			sample_m = torch.abs(sample_c)
			sample_re = sample_m*torch.cos(e_rand).unsqueeze(0)
			sample_im = sample_m*torch.sin(e_rand).unsqueeze(0)
			return torch.cat([sample_re, sample_im], axis=0)
		else:
			return sample

class DownsampleZeroFill(torch.nn.Module):
	"""
	linear downsampler based sz by sz grid of fixed width 
	adapted from operator/Downsampler in csi_sr branch of iterative_reconstruction_networks repo
	key difference: use gramian (X^TXb) as forward operator, 
	fills zeros between samples
	"""
	
	def __init__(self, sz=5, H=32, W=32, vect_bool=False, device="cpu"):
		super(DownsampleZeroFill, self).__init__()
		self.vect_bool = vect_bool # indicates vectorized inputs
		self.sz, self.H, self.W = sz, H, W

		# build downsampling matrix, X
		# TODO: take downsampling matrix as input
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		X = torch.zeros((sz*sz, H*W)) # template
		i = 0
		for val_y in grid_temp.view(-1):
			for val_x in grid_temp.view(-1):
				idx_x = int(torch.round((W-1) * (val_x + 1) / 2))
				idx_y = int(torch.round((H-1) * (val_y + 1) / 2))
				j = idx_y*H+idx_x
				X[i,j] = 1.0
				i += 1
		self.X_down = X.to(device)

	def forward(self, x):
		# assume re/im channels on axis=1
		# print(f"forward input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		# x = x.view(x.size(0), self.H*self.W) if not self.vect_bool else x # account for channels (axs=0)
		x = torch.reshape(x, (sz_in[0], self.H*self.W)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down.T).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down.T).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.sz, self.sz)
		return x

	def adjoint(self, x):
		# assume re/im channels on axis=1
		# print(f"adjoint input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		# x = x.view(x.size(0), self.sz*self.sz) if not self.vect_bool else x # account for channels (axs=0)
		x = torch.reshape(x, (sz_in[0], self.sz*self.sz)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.H, self.W)
		return x 

	def gramian(self, x):
		return self.adjoint(self.forward(x))

	def __call__(self, x):
		# print(f"gramian input: {x.size()}")
		return (x, self.gramian(x))

class Downsample(torch.nn.Module):
	"""
	linear downsampler based sz by sz grid of fixed width 
	adapted from operator/Downsampler in csi_sr branch of iterative_reconstruction_networks repo
	key difference: use gramian (X^TXb) as forward operator, 
	default: fills zeros between samples.
	optionally: bilinear interpolation between samples.
	"""
	
	def __init__(self, sz=5, H=32, W=32, vect_bool=False, interpolate=False, mode=None, device="cpu"):
		super(Downsample, self).__init__()
		self.vect_bool = vect_bool # indicates vectorized inputs
		self.sz, self.H, self.W = sz, H, W
		self.interpolate = interpolate
		self.mode = "bilinear" if mode is None else "bicubic" 

		# build downsampling matrix, X
		# TODO: take downsampling matrix as input
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		X = torch.zeros((sz*sz, H*W)) # template
		i = 0
		for val_y in grid_temp.view(-1):
			for val_x in grid_temp.view(-1):
				idx_x = int(torch.round((W-1) * (val_x + 1) / 2))
				idx_y = int(torch.round((H-1) * (val_y + 1) / 2))
				j = idx_y*H+idx_x
				X[i,j] = 1.0
				i += 1
		self.X_down = X.to(device)

	def forward(self, x):
		# assume re/im channels on axis=1
		# print(f"forward input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], self.H*self.W)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down.T).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down.T).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.sz, self.sz)
		return x

	def adjoint(self, x):
		# assume re/im channels on axis=1
		# print(f"adjoint input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], self.sz*self.sz)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.H, self.W)
		return x 

	def gramian(self, x):
		return self.adjoint(self.forward(x))

	def __call__(self, x):
		# print(f"gramian input: {x.size()}")
		if self.interpolate:	
			y = self.forward(x)
			y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(self.H,self.W), mode=self.mode).squeeze()
		else:
			y = self.gramian(x)
		return (x, y)

class DownsampleSingleAxis(torch.nn.Module):
	"""
	linear downsampler to sz along single axis
	adapted from operator/Downsampler in csi_sr branch of iterative_reconstruction_networks repo
	key difference: use gramian (X^TXb) as forward operator, 
	default: fills zeros between samples.
	optionally: bilinear interpolation between samples.
	"""
	
	def __init__(self, sz=5, H=32, W=32, vect_bool=False, interpolate=False, mode=None, device="cpu"):
		super(DownsampleSingleAxis, self).__init__()
		self.vect_bool = vect_bool # indicates vectorized inputs
		self.sz, self.H, self.W = sz, H, W
		self.interpolate = interpolate
		self.mode = "bilinear" if mode is None else "bicubic" 

		# build downsampling matrix, X
		# TODO: take downsampling matrix as input
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		X = torch.zeros((H*sz, H*W)) # template
		i = 0
		for idx_y in range(H):
			for val_x in grid_temp.view(-1):
				idx_x = int(torch.round((W-1) * (val_x + 1) / 2))
				j = idx_y*W+idx_x
				X[i,j] = 1.0
				i += 1
				# print(f"({i}, {j})")
		self.X_down = X.to(device)

	def forward(self, x):
		# assume re/im channels on axis=1
		# print(f"forward input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], self.H*self.W)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down.T).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down.T).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.H, self.sz)
		return x

	def adjoint(self, x):
		# assume re/im channels on axis=1
		# print(f"adjoint input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], self.H*self.sz)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], self.X_down).unsqueeze(0)
		x_im = torch.matmul(x[1,:], self.X_down).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.H, self.W)
		return x 

	def gramian(self, x):
		return self.adjoint(self.forward(x))

	def __call__(self, x):
		# print(f"gramian input: {x.size()}")
		if self.interpolate:	
			y = self.forward(x)
			y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(self.H,self.W), mode=self.mode).squeeze()
		else:
			y = self.gramian(x)
		return (x, y)

class DownsampleMultiscale(torch.nn.Module):
	"""
	linear downsampler based on sz by sz grid of fixed width 
	return grids of sz_lo, sz_hi s.t. sz_lo < sz_hi
	adapted from operator/Downsampler in csi_sr branch of iterative_reconstruction_networks repo
	key difference: use gramian (X^TXb) as forward operator, 
	default: fills zeros between samples.
	optionally: bilinear interpolation between samples.
	"""
	
	def __init__(self, sz_lo=8, sz_hi=16, H=32, W=32, vect_bool=False, interpolate=False, mode=None, device="cpu"):
		super(DownsampleMultiscale, self).__init__()
		self.vect_bool = vect_bool # indicates vectorized inputs
		self.sz_lo, self.sz_hi, self.H, self.W = sz_lo, sz_hi, H, W
		self.interpolate = interpolate
		self.mode = "bilinear" if mode is None else "bicubic" 

		# build downsampling matrix, X
		# TODO: take downsampling matrix as input
		# if sz % 2 == 0: # even size
		# 	sz_half = int(sz/2)
		# 	scale = sz_half - 0.5
		# 	grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		# else: # odd size
		# 	sz_half = int(sz/2)
		# 	scale = sz_half 
		# 	grid_temp = (torch.arange(sz) - sz_half) / scale
		# X = torch.zeros((sz*sz, H*W)) # template
		# i = 0
		# for val_y in grid_temp.view(-1):
		# 	for val_x in grid_temp.view(-1):
		# 		idx_x = int(torch.round((W-1) * (val_x + 1) / 2))
		# 		idx_y = int(torch.round((H-1) * (val_y + 1) / 2))
		# 		j = idx_y*H+idx_x
		# 		X[i,j] = 1.0
		# 		i += 1
		# self.X_down = X.to(device)
		self.X_down_lo = make_downsampling_mat(self.sz_lo, H, W).to(device)
		self.X_down_hi = make_downsampling_mat(self.sz_hi, H, W).to(device)

	def forward(self, x, X_down, sz):
		# assume re/im channels on axis=1
		# print(f"forward input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], self.H*self.W)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], X_down.T).unsqueeze(0)
		x_im = torch.matmul(x[1,:], X_down.T).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], sz, sz)
		return x

	def adjoint(self, x, X_down, sz):
		# assume re/im channels on axis=1
		# print(f"adjoint input: {x.size()} - self.X_down.T.size(): {self.X_down.T.size()}")
		sz_in = x.size()
		x = torch.reshape(x, (sz_in[0], sz*sz)) if not self.vect_bool else x # account for channels (axs=0)
		x_re = torch.matmul(x[0,:], X_down).unsqueeze(0)
		x_im = torch.matmul(x[1,:], X_down).unsqueeze(0)
		x = torch.cat([x_re, x_im], axis=0).view(sz_in[0], self.H, self.W)
		return x 

	def gramian(self, x, X_down, sz):
		return self.adjoint(self.forward(x, X_down, sz), X_down, sz)

	def __call__(self, x):
		"""
		run downsampling at two scales, sz_lo and sz_hi
		"""
		y_lo = self.forward(x, self.X_down_lo, self.sz_lo)
		y_hi = self.forward(x, self.X_down_hi, self.sz_hi)
		if self.interpolate:	
			y_lo = torch.nn.functional.interpolate(y_lo.unsqueeze(0), size=(self.sz_hi,self.sz_hi), mode=self.mode).squeeze()
			# y_hi = torch.nn.functional.interpolate(y_hi.unsqueeze(0), size=(self.H,self.W), mode=self.mode).squeeze()
			
		return (y_lo, y_hi)

def make_downsampling_mat(sz, H, W):
	""" 
	returns fixed-grid downsampling matrix taking H x W -> sz x sz
	"""
	if sz % 2 == 0: # even size
		sz_half = int(sz/2)
		scale = sz_half - 0.5
		grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
	else: # odd size
		sz_half = int(sz/2)
		scale = sz_half 
		grid_temp = (torch.arange(sz) - sz_half) / scale
	X = torch.zeros((sz*sz, H*W)) # template
	i = 0
	for val_y in grid_temp.view(-1):
		for val_x in grid_temp.view(-1):
			idx_x = int(torch.round((W-1) * (val_x + 1) / 2))
			idx_y = int(torch.round((H-1) * (val_y + 1) / 2))
			j = idx_y*H+idx_x
			X[i,j] = 1.0
			i += 1
	return X