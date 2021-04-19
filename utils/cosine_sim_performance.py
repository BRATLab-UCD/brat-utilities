import numpy as np

def cosine_similarity(A, B, n_del=32, n_ang=32):
    """ 
    -> return average cosine similarity between elements in estimate and ground truth  (A and B)
    A.shape = B.shape = (n_samples, T, n_del, n_ang)
    """
    N = A.shape[0]
    rho = 0
    for i in range(N):
        for t in range(T):
            for i_ang in range(n_ang):
                A_i = A[i,t,:,i_ang]
                B_i = B[i,t,:,i_ang]
                AB_dot = np.abs(np.dot(A_i, np.conj(B_i)))
                A_norm = np.sqrt(np.sum(A_i*np.conj(A_i)))
                B_norm = np.sqrt(np.sum(B_i*np.conj(B_i)))
                rho += AB_dot / (A_norm*B_norm) / (T*n_ang*N)
    return np.real(rho)

if __name__ == "__main__":
    N = 100
    n_del = 32
    n_ang = 32
    T = 10
    A = np.random.normal(size=(N,T,n_del,n_ang)) + 1j*np.random.normal(size=(N,T,n_del,n_ang))
    sigma_list = [0.1,1.0,10.0]
    print(f"--- Testing: cosine_similarity(A,B) for B = A + CN(0,sigma) ---")
    A_rho = cosine_similarity(A,A,n_del=n_del,n_ang=n_ang)
    print(f"sigma={0:2.1E} -> A_rho: {A_rho}")
    for sigma in sigma_list:
        B = A + np.random.normal(0,sigma,size=(N,T,n_del,n_ang)) + 1j*np.random.normal(0,sigma,size=(N,T,n_del,n_ang))
        AB_rho = cosine_similarity(A,B,n_del=n_del,n_ang=n_ang)
        print(f"sigma={sigma:2.1E} -> AB_rho: {AB_rho}")