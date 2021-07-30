import torch

def b_trace(b_mat, eye):	 
    """
    apply trace to each matrix/batch in tensor

	parameters:
    -> b_mat = (B, N, N) B batches of N x N matrices which
            to take the trace of
    -> eye = (B, N, N) batched identity matrix; allocated
            before calling to avoid creating new tensors 
            at runtime

    output:
    -> out = (B,) trace of all B matrices in tensor
    """
    b_diag = torch.mul(eye, b_mat)
    b_trace = torch.sum(b_diag.view(b_diag.size(0),-1), axis=1)
    return b_trace


def zero_forcing(H, P, eye=None):
	"""
	return zero forcing (ZF) precoder and power constant based on channel 
	response and power constraint. Precoder defs from Sohrabi, Foad, and 
	Attiah, Kareem M., and Yu, We, "Deep Learning for Distributed Channel
	Feedback and Multiuser Precoding in FDD Massive MIMO." 2020-07.
	
	torch implementation
	"""
	if len(H.size()) == 2:
		H_inv = torch.linalg.inv(torch.matmul(H, torch.conj(H).T))
		V = torch.matmul(np.conj(H.T), H_inv) # hermitian
		gamma = 1 / torch.sqrt(P*torch.trace(torch.matmul(V, torch.conj(V.T)))) # get power constraint
	if len(H.size()) == 3:
		assert(not eye is None)
		H_inv = torch.linalg.inv(torch.matmul(H, torch.conj(torch.transpose(H, 1, 2))))
		V = torch.matmul(torch.conj(torch.transpose(H, 1, 2)), H_inv) # hermitian
		gamma = 1 / torch.sqrt(P*b_trace(torch.matmul(V, torch.conj(torch.transpose(V, 1, 2))), eye)) # get power constraint
	return [gamma, V]


def maximal_ratio_transmission(H, P, eye=None):
	"""
	return maximal ratio transmission (MRT) precoder and power constant based on channel 
	response and power constraint. Precoder defs from Sohrabi, Foad, and 
	Attiah, Kareem M., and Yu, We, "Deep Learning for Distributed Channel
	Feedback and Multiuser Precoding in FDD Massive MIMO." 2020-07.

	torch implementation

	MRT precoder def:
	$ \mathbf V = \gamma_{\text{ZF}}\mathbf H^H$

	With power constraint:
	$ \text{Tr}(\mathbf V\mathbf V^H) \leq P$

	parameters:
	-> H = (B x M x K) or (M x K) channel response
	-> P = power constraint
	"""
	if len(H.size()) == 2:
		V = torch.conj(H.T) # hermitian
		gamma = 1 / torch.sqrt(P*torch.trace(torch.matmul(V, torch.conj(V.T)))) # get power constraint
	if len(H.size()) == 3:
		assert(not eye is None)
		V = torch.conj(torch.transpose(H,1,2)) # hermitian
		foo = torch.matmul(V, torch.conj(torch.transpose(V, 1, 2)))
		gamma = 1 / torch.sqrt(P*b_trace(foo, eye)) # get power constraint
	return [gamma, V]


def sum_rate(H, V, sigma):
	"""
	return user rate given channel, precoder, and noise level. Defs from Sohrabi, Foad, and 
	Attiah, Kareem M., and Yu, We, "Deep Learning for Distributed Channel
	Feedback and Multiuser Precoding in FDD Massive MIMO." 2020-07.

	torch implementation

	rate def for user k:
	$ R_k = \log_2(1 + \frac{|\mathbf h_k^H \mathbf v_k|^2}{\sum_{j\neq k}|\mathbf h_j^H \mathbf v_j|^2 + \sigma^2})

	sum rate def for all users:
	$ R = \sum_k^K R_k$ 

	parameters:
	-> H = (M x K) or (B x M x K) channel response
	-> V = (K x M) precoding matrix
	-> sigma = additive noise power
	"""
	if len(H.size()) == 2:
		M, K = H.size()
		rate = 0 
		signal_list = []
		interf_list = []
	elif len(H.size()) == 3:
		B, M, K = H.size() # batch size
		rate = torch.zeros(B,1).squeeze()
		signal_list = torch.zeros(B,K)
		interf_list = torch.zeros(B,K)
	for k in range(K):
		if len(H.size()) == 2:
			h_k = torch.conj(H[:,k].T)
			r_user = torch.abs(torch.dot(V[k,:], h_k))**2
			if k == 0:
				V_temp = V[k+1:,:]
			elif k == K-1:
				V_temp = V[:k,:]
			else:
				V_temp = torch.cat([V[:k,:], V[k+1:,:]])
			r_int = torch.sum(torch.abs(torch.matmul(V_temp, h_k))**2) # interference term
			rate += torch.log2(1 + (r_user / (r_int + sigma)))
			signal_list.append(r_user.numpy())
			interf_list.append(r_int.numpy())
		elif len(H.size()) == 3:
			h_k = torch.conj(H[:,:,k])
			r_user = torch.abs(torch.sum(torch.mul(V[:,k,:], h_k), axis=1))**2
			if K > 1:
				if k == 0:
					V_temp = V[:,k+1:,:]
				elif k == K-1:
					V_temp = V[:,:k,:]
				else:
					V_temp = torch.cat([V[:,:k,:], V[:,k+1:,:]], axis=1)
				r_int = torch.abs(torch.sum(torch.matmul(V_temp, h_k.unsqueeze(2)), axis=1))**2 # interference term
			else:
				r_int =  torch.zeros(r_user.size())
			rate = rate + torch.log2(1 + (r_user / (r_int.squeeze() + sigma)))
			signal_list[:,k] = r_user
			interf_list[:,k] = r_int.squeeze()
	return [signal_list, interf_list, rate]