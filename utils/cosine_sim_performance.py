import numpy as np

def cosine_similarity(A, B, pow_diff_T=None):
    """ 
    -> return average cosine similarity between elements in estimate and ground truth  (A and B)
    A.shape = B.shape = (n_samples, T, n_freq, n_ang)
    """
    N, T, n_freq, n_ant = A.shape
    rho_trunc = 0
    # rho_all = 0
    for i in range(N):
        for t in range(T):
            # assume power uniformly distributed among angles
            # pow_diff = 0 if type(pow_diff_T) == type(None) else pow_diff_T[i,t,0] / n_ant
            for i_freq in range(n_freq):
                A_i = A[i,t,i_freq,:]
                B_i = B[i,t,i_freq,:]
                AB_dot = np.abs(np.dot(A_i, np.conj(B_i)))
                A_norm = np.sqrt(np.sum(A_i*np.conj(A_i)))
                B_norm = np.sqrt(np.sum(B_i*np.conj(B_i))) 
                rho_trunc += AB_dot / (A_norm*B_norm) / (T*n_ant*N)
                # rho_all += AB_dot / (A_norm*(B_norm+pow_diff)) / (T*n_ant*N)
    # return [np.real(rho_trunc), np.real(rho_all)]
    return np.real(rho_trunc)

if __name__ == "__main__":
    N = 100
    n_del = 32
    n_ang = 32
    T = 10
    A = np.random.normal(size=(N,T,n_del,n_ang)) + 1j*np.random.normal(size=(N,T,n_del,n_ang))
    # pow_diff = np.random.normal(0,0.1,size=(N,T))
    sigma_list = [0.1,1.0,10.0]
    print(f"--- Testing: cosine_similarity(A,B) for B = A + CN(0,sigma) ---")
    A_rho = cosine_similarity(A,A)
    print(f"sigma={0:2.1E} -> cos(A,A): {A_rho}")
    for sigma in sigma_list:
        B = A + np.random.normal(0,sigma,size=(N,T,n_del,n_ang)) + 1j*np.random.normal(0,sigma,size=(N,T,n_del,n_ang))
        # AB_rho, AB_rho_all = cosine_similarity(A,B,pow_diff_T=pow_diff)
        # print(f"sigma={sigma:2.1E} -> cos_truncate(A,B): {AB_rho} - cos_all(A,B): {AB_rho_all}")
        AB_rho = cosine_similarity(A,B)
        print(f"sigma={sigma:2.1E} -> cos_truncate(A,B): {AB_rho}")