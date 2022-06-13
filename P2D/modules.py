import numpy as np
import torch
from torch import nn
# from pytorch_wavelets import DWTForward, DWTInverse

class P2D:
	def __init__(self, M, N, N_t):
		self.M = M
		self.N = N
		self.N_t = N_t
		self.make_downsample_matrix(M, N)

	def make_downsample_matrix(self, sz, N):
		""" 
		returns fixed-grid downsampling matrix taking N -> sz
		"""
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		X = torch.zeros((sz, N)) # template
		i = 0
		for val_y in grid_temp.view(-1):
			j = int(torch.round((N-1) * (val_y + 1) / 2))
			X[i,j] = 1.0
			i += 1
		self.P = X

	def fit(self, delta=0.5):
		"""
		return coefficients to predict truncated delay domain based on downsampled pilots:
		\bar{h}_d = Q_pre \bar{h}_f
			  = (Q_t^TQ_t)^{-1}Q_t \bar{h}_f
		"""
		P = self.P
		M, N, N_t = self.M, self.N, self.N_t
	
		I = np.eye(N)
		F = np.fft.fft(I) # get fft matrix

		Q = np.dot(P, F) 
		Q = Q[:,:N_t] # truncate
		QT_Q = np.dot(np.conj(Q).T, Q)
		Q_diag = np.diag(np.diag(QT_Q))
		if delta != 0:
			QT_Q = (QT_Q - Q_diag) / (1 + delta) + Q_diag
		Q_inv = np.linalg.inv(QT_Q)
		Q_pre = np.dot(Q_inv, np.conj(Q).T)

		self.Q_pre = Q_pre

	def predict(self, x):
		return np.dot(x, self.Q_pre.T)

class P2D_Diag:
	def __init__(self, M, N, N_t, D):
		self.M = M
		self.N = N
		self.N_t = N_t
		self.D = D

		self.make_downsample_matrix(M, N, D)

	def make_downsample_matrix(self, sz, N, D):
		""" 
		returns D fixed-grid downsampling matrces that take N -> sz
		"""
		N = N - D
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		# print(grid_temp)
		X = torch.zeros((D, sz, N+D)) # template
		j = 0
		for val_y in grid_temp.view(-1):
			k = int(torch.round(N * (val_y + 1) / 2))
			for i in range(D):
				# print(f"X[{i},{j+i},{k+i}] with j={j}, k={k}")
				X[i,j,k+i] = 1.0
			j += 1
		self.P = X

	def fit(self, delta=0.5):
		"""
		return coefficients to predict truncated delay domain based on downsampled pilots:
		\bar{h}_d = Q_pre \bar{h}_f
			  = (Q_t^TQ_t)^{-1}Q_t \bar{h}_f
		"""
		P = self.P
		D, M, N, N_t = self.D, self.M, self.N, self.N_t
	
		I = np.eye(N)
		F = np.fft.fft(I) # get fft matrix

		Q_pre = np.zeros((D, M, N_t), dtype="complex")
		for i in range(D):
			P_i = P[i,:,:]
			Q = np.dot(P_i, F) 
			Q = Q[:,:N_t] # truncate
			QT_Q = np.dot(np.conj(Q).T, Q)
			Q_diag = np.diag(np.diag(QT_Q))
			if delta != 0:
				QT_Q = (QT_Q - Q_diag) / (1 + delta) + Q_diag
			Q_inv = np.linalg.inv(QT_Q)
			Q_pre[i,:,:] = np.dot(Q_inv, np.conj(Q).T).T

		self.Q_pre = Q_pre

	def downsample(self, x):
		D = self.D
		N_s, N_f = x.shape
		y = np.zeros((N_s, self.M), dtype="complex")
		for i in range(N_s):
			d = i % D
			P = self.P[d,:,:]
			y[i,:] = np.dot(x[i,:], P.T)
			# y[:,i,:] = np.expand_dims(np.dot(P, x[:,i,:]), 1)
		return y

	def predict(self, x):
		D = self.D
		N_s, M_f = x.shape
		y = np.zeros((N_s, self.N_t), dtype="complex")
		for i in range(N_s):
			d = i % D
			Q_pre = self.Q_pre[d,:,:]
			y[i,:] = np.dot(x[i,:], Q_pre)
		return y	

class P2AD:
	# pilots to angular-delay
	def __init__(self, M, N, M_b, N_b, N_t):
		self.M = M # downsample size (frequency)
		self.N = N # original size (frequency)
		self.M_b = M_b # downsample size (spatial)
		self.N_b = N_b # original size (spatial)
		self.N_t = N_t # truncation value (delay axis)
		self.P = self.make_downsample_matrix(M, N)
		self.D = self.make_downsample_matrix(M_b, N_b)

	def make_downsample_matrix(self, sz, N):
		""" 
		returns fixed-grid downsampling matrix taking N -> sz
		"""
		if sz % 2 == 0: # even size
			sz_half = int(sz/2)
			scale = sz_half - 0.5
			grid_temp = (torch.arange(sz) + 0.5 - sz_half) / scale
		else: # odd size
			sz_half = int(sz/2)
			scale = sz_half 
			grid_temp = (torch.arange(sz) - sz_half) / scale
		X = torch.zeros((sz, N)) # template
		i = 0
		for val_y in grid_temp.view(-1):
			j = int(torch.round((N-1) * (val_y + 1) / 2))
			X[i,j] = 1.0
			i += 1
		return X

	def fit(self, delta=0.5, delta_ang=0.5):
		"""
		return coefficients to predict truncated delay domain based on downsampled pilots:
		\bar{h}_d = Q_pre \bar{h}_f
			  = (Q_t^TQ_t)^{-1}Q_t \bar{h}_f
		"""
		P, D = self.P, self.D
		M, N, N_t = self.M, self.N, self.N_t
		M_b, N_b = self.M_b, self.N_b

		# get R matrix
		I = np.eye(N_b)
		F = np.fft.ifft(I) # get ifft matrix
		R = np.dot(D, F) 

		# pseudoinverse of D w/ ODIR regularization
		RT_R = np.dot(np.conj(R).T, R)
		R_diag = np.diag(np.diag(RT_R))
		if delta_ang != 0:
			RT_R = (RT_R - R_diag) / (1 + delta_ang) + R_diag
		R_inv = np.linalg.inv(RT_R)
		R_pre = np.dot(R_inv, np.conj(R).T)

		self.R_pre = R_pre
	
		# get Q matrix
		I = np.eye(N)
		F = np.fft.fft(I) # get fft matrix
		Q = np.dot(P, F) 
		Q = Q[:,:N_t] # truncate

		# pseudo inverse of Q w/ ODIR regularization
		QT_Q = np.dot(np.conj(Q).T, Q)
		Q_diag = np.diag(np.diag(QT_Q))
		if delta != 0:
			QT_Q = (QT_Q - Q_diag) / (1 + delta) + Q_diag
		Q_inv = np.linalg.inv(QT_Q)
		Q_pre = np.dot(Q_inv, np.conj(Q).T)

		self.Q_pre = Q_pre

	def predict(self, x):
		# delay, then angle
		x = np.dot(x, self.Q_pre.T)
		x = np.transpose(x,(1,0))
		x = np.dot(x, self.R_pre.T)
		x = np.transpose(x,(1,0))
		return x
