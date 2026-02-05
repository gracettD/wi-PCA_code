import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize


class Estimator:
    """
    Attributes
    Y : panel observation
    W : missing pattern
    prob: missing probability (default = None)
    """
    def __init__(self, Y, W, prob=None):
        self.Y = Y
        self.W = W 
        self.prob = prob
    

    def PCA(self, k): 
        """
        The PCA method in Xiong and Pelger (2022)
        """ 
        W = self.W
        Y = self.Y
        N, T = Y.shape
        # Estimate the covariance matrix
        Q = W @ W.T
        Yobs = Y * W
        Sigma = (Yobs @ Yobs.T) / Q
        # Estimate loadings
        e_vals, e_vecs = np.linalg.eig(Sigma/N)
        sorted_indices = np.argsort(e_vals)
        e_vecs = np.real(e_vecs)
        Lam_hat = e_vecs[:,sorted_indices[:-k-1:-1]]
        Lam_hat = Lam_hat * np.sqrt(N)
        # Estimate factors
        F_hat = np.zeros((T,k))
        for t in range (T):
            w_diag = np.diag(W[:,t])
            F_hat[t, :] = (inv(Lam_hat.T @ w_diag @ Lam_hat) @ (Lam_hat.T @ w_diag @ Y[:,t])).T           
        C_hat = Lam_hat @ F_hat.T
        return C_hat


    def block_PCA(self, k):
        """
        Block-PCA estimator in Xu (2017)
        """
        W = self.W
        Y = self.Y
        N, T = Y.shape 
        # Estimate factors 
        control_group = [i for i in range (N) if sum(W[i,:])==T]
        Y_full = Y[control_group, :]
        Sigma = (Y_full.T @ Y_full) / len(control_group)
        e_vals, e_vecs = np.linalg.eig(Sigma/T)
        sorted_indices = np.argsort(e_vals)
        e_vecs = np.real(e_vecs)
        F_hat = e_vecs[:,sorted_indices[:-k-1:-1]]
        F_hat = F_hat * np.sqrt(T)
        # Estimate loadings
        Lam_hat = np.zeros((N, k))
        for i in range (N):
            w_diag = np.diag(W[i,:])
            Lam_hat[i, :] = (inv(F_hat.T @ w_diag @ F_hat) @ (F_hat.T @ w_diag @ (Y[i,:].reshape(T,1)))).T           
        C_hat = Lam_hat @ F_hat.T
        return C_hat

    def causalPCA_known(self, k):
        """
        wi-PCA estimator with known observation probability
        """
        W = self.W
        Y = self.Y
        prob = self.prob
        N, T = Y.shape
        # fixed effects
        xi_hat = (np.sum(W*Y/prob, axis=0)/np.sum(W/prob, axis=0)).reshape(T,1)
        alpha_hat = (np.sum(W*(Y-np.ones((N,1))@(xi_hat.T)), axis=1)/np.sum(W, axis=1)).reshape(N,1)
        # factor structure
        tildeY = Y - alpha_hat@np.ones((1,T)) - np.ones((N,1))@xi_hat.T
        C_hat = Estimator(Y=tildeY, W=W).PCA(k)
        C_hat += alpha_hat@np.ones((1,T)) + np.ones((N,1))@xi_hat.T  
        return C_hat
    
    
    def causalPCA_unknown_1(self, k):
        """
        wi-PCA estimator with unknown observation probability and short-term dependency in missingness
        """
        W = self.W
        Y = self.Y
        N, T = Y.shape
        # fixed effects
        inv_Wi = (1/np.mean(W, axis=1)).reshape(N,1)
        xi_hat = ((inv_Wi.T@(W*Y)) / (inv_Wi.T@W)).reshape(T,1)
        alpha_hat = (np.sum(W*(Y-np.ones((N,1))@(xi_hat.T)), axis=1)/np.sum(W, axis=1)).reshape(N,1)
        # factor structure
        tildeY = Y  - alpha_hat@np.ones((1,T)) - np.ones((N,1))@xi_hat.T
        C_hat = Estimator(Y=tildeY, W=W).PCA(k)
        C_hat += alpha_hat@np.ones((1,T)) + np.ones((N,1))@xi_hat.T  
        return C_hat
    
    
    def causalPCA_unknown_2(self, k):
        """
        wi-PCA estimator with unknown observation probability and monotone missingness
        """
        W = self.W
        Y = self.Y
        N, T = Y.shape
        # fixed effects
        control_group = [i for i in range (N) if sum(W[i,:])==T]
        Y_full = Y[control_group, :]
        xi_hat = (np.mean(Y_full, axis=0)).reshape(T,1)
        alpha_hat = (np.sum(W*(Y-np.ones((N,1))@(xi_hat.T)), axis=1)/np.sum(W, axis=1)).reshape(N,1)
        # factor structure
        tildeY = Y  - alpha_hat@np.ones((1,T)) - np.ones((N,1))@xi_hat.T
        C_hat = Estimator(Y=tildeY, W=W).PCA(k)
        C_hat += alpha_hat@np.ones((1,T)) + np.ones((N,1))@xi_hat.T  
        return C_hat
    
  
  
def did_objective(params, Y, W):
    N,T = Y.shape
    alpha = params[:N]
    beta = params[N:]
    res = Y - alpha.reshape(-1,1)@np.ones((1,T)) - np.ones((N,1))@beta.reshape(1,-1)
    return np.sum((res**2) * W)


def did_estimator(Y, W):
    """
    TWFE estimator
    """
    N,T = Y.shape
    x0 = np.zeros(N+T)
    result = minimize(did_objective, x0, args=(Y,W))
    params = result.x
    alpha = params[:N]
    beta = params[N:]
    Y_hat = alpha.reshape(-1,1) @ np.ones((1,T)) + np.ones((N,1)) @ beta.reshape(1,-1)
    return Y_hat


