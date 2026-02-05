# This file generates Table 2

import numpy as np
from numpy import *
from Estimator import Estimator, did_estimator


# generate data for simulation
def data(sigma_eps, stationary, N=100, T=100):  
    k = 1; mu = 1
    F = np.random.randn(T,k); Lam = np.random.randn(N,k)
    eps = np.random.randn(N,T) * sigma_eps
    alpha = np.random.randn(N,1)
    if stationary == True:  # stationary time fixed effects
        xi = np.random.randn(T,1)   
    else:  # non-stationary time fixed effects
        xi = np.array([0.05*t+np.random.randn(1) for t in range (T)]).reshape(T,1)
    xi = xi - np.mean(xi)
    # generate common component
    C = mu*np.ones((N,T)) + alpha@np.ones((1,T)) + np.ones((N,1))@xi.T + Lam@(F.T)    
    Y = C + eps
    return Y, C, alpha, xi


def relativeMSE(W, C, C_hat):
    MSE_all = np.sum((C-C_hat)**2) / np.sum(C**2)
    MSE_obs = np.sum(((C-C_hat)*W)**2) / np.sum((C*W)**2)
    MSE_miss = np.sum(((C-C_hat)*(1-W))**2) / np.sum((C*(1-W))**2)
    return MSE_all, MSE_obs, MSE_miss


N, T = 100, 100
Simutime = 200
max_factor = 3
# sigma_eps_set = [1, 2, 3]
sigma_eps_set = [2]


# 1. missing-at-random
print("--------------------------")
print("Missing-at-random")

def W_random(p, N=100, T=100):  # generate observation pattern
    prob = p * np.ones((N,T))
    W = np.random.binomial(size=(N,T), n=1, p=p) 
    return W, prob

# p_set = [0.4, 0.6, 0.8]
p_set = [0.8]
for p in p_set:
    for sigma_eps in sigma_eps_set: 
        print(f"p = {p}, sigma_eps = {sigma_eps}")
        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = np.zeros(3),np.zeros(3),np.zeros((max_factor,3)),np.zeros((max_factor,3)),np.zeros(3)

        # stationary
        print("--------------------------")
        print("Stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=True)
            W, prob = W_random(p=p)
            while any(W @ W.T == 0):
                W, prob = W_random(p=p)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_1(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &- &- &- &{MSE_did[line]:.3f}'+'\\'+'\\')


        # non-stationary
        print("--------------------------")
        print("Non-stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=False)
            W, prob = W_random(p=p)
            while any(W @ W.T == 0):
                W, prob = W_random(p=p)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_1(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &- &- &- &{MSE_did[line]:.3f}'+'\\'+'\\')
    

# 2. simultaneous treatment adoption
print("--------------------------")
print("Simultaneous treatment adoption")

def W_simultaneous(periods, N=100, T=100):  # generate observation pattern
    prob = np.ones((N,T))
    W = np.ones((N,T))
    treat_time = int(periods * T)
    for i in range (N):
        prob[i, treat_time:] = 0.5
        flag = np.random.binomial(n=1, p=0.5)
        if flag == 0:
            W[i, treat_time:] = 0
    return W, prob

# p_set = [0.4,0.5,0.6]
p_set = [0.4]
for p in p_set:
    for sigma_eps in sigma_eps_set: 
        print(f"p = {p}, sigma_eps = {sigma_eps}")
        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = np.zeros(3),np.zeros(3),np.zeros((max_factor,3)),np.zeros((max_factor,3)),np.zeros(3)

        # stationary
        print("--------------------------")
        print("Stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=True)
            W, prob = W_simultaneous(periods=p)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_2(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # Block-PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).block_PCA(k=factor_number)
                MSE_block[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &{MSE_block[0,line]:.3f} &{MSE_block[1,line]:.3f} &{MSE_block[2,line]:.3f} &{MSE_did[line]:.3f}'+'\\'+'\\')


        # non-stationary
        print("--------------------------")
        print("Non-stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=False)
            W, prob = W_simultaneous(periods=p)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_2(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # Block-PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).block_PCA(k=factor_number)
                MSE_block[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &{MSE_block[0,line]:.3f} &{MSE_block[1,line]:.3f} &{MSE_block[2,line]:.3f} &{MSE_did[line]:.3f}'+'\\'+'\\')



# 3. staggered treatment adoption
print("--------------------------")
print("Staggered treatment adoption")

def W_staggered(p, xi, alpha, N=100, T=100):   # generate observation pattern
    prob = np.ones((N,T))
    W = np.ones((N,T))
    beginning = int(0.1 * T)
    for i in range (N):
        for t in range (beginning, T):
            if abs(xi[t]*alpha[i]) > 2.5:
                prob[i,t:] = p * prob[i,t:]
                W[i,t] = np.random.binomial(1, p=p)
                if W[i,t] == 0:
                    W[i,t+1:] = np.zeros(T-t-1)
                    break
    return W, prob

# p_set = [0.5, 0.7, 0.9]
p_set = [0.9]
for p in p_set:
    for sigma_eps in sigma_eps_set: 
        print(f"p = {p}, sigma_eps = {sigma_eps}")
        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = np.zeros(3),np.zeros(3),np.zeros((max_factor,3)),np.zeros((max_factor,3)),np.zeros(3)
      
        # stationary
        print("--------------------------")
        print("Stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=True)
            W, prob = W_staggered(p=p, xi=xi, alpha=alpha)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_2(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # Block-PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).block_PCA(k=factor_number)
                MSE_block[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &{MSE_block[0,line]:.3f} &{MSE_block[1,line]:.3f} &{MSE_block[2,line]:.3f} &{MSE_did[line]:.3f}'+'\\'+'\\')


        # non-stationary
        print("--------------------------")
        print("Non-stationary")
        for s in range (Simutime):
            Y, C, alpha, xi = data(sigma_eps=sigma_eps, stationary=False)
            W, prob = W_staggered(p=p, xi=xi, alpha=alpha)
            
            # wi-PCA with known probability
            C_hat = Estimator(Y, W, prob).causalPCA_known(1)
            MSE_causal_1 += relativeMSE(W, C, C_hat) 

            # wi-PCA with unknown probability
            C_hat = Estimator(Y, W).causalPCA_unknown_2(1)
            MSE_causal_2 += relativeMSE(W, C, C_hat)
            
            # PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).PCA(k=factor_number)
                MSE_xp[f] += relativeMSE(W, C, C_hat)

            # Block-PCA
            for f in range (max_factor):
                factor_number = f+1
                C_hat = Estimator(Y, W).block_PCA(k=factor_number)
                MSE_block[f] += relativeMSE(W, C, C_hat)

            # TWFE
            C_hat = did_estimator(Y, W)
            MSE_did += relativeMSE(W, C, C_hat)

        MSE_causal_1, MSE_causal_2, MSE_xp, MSE_block, MSE_did = MSE_causal_1/Simutime, MSE_causal_2/Simutime, MSE_xp/Simutime, MSE_block/Simutime, MSE_did/Simutime
        for line in [1,2,0]:
            if line == 1:
                flag = '& obs'
            elif line == 2:
                flag = '&& miss'
            elif line == 0:
                flag = '&& all'
            print(flag + f'&{MSE_causal_1[line]:.3f} &{MSE_causal_2[line]:.3f} &{MSE_xp[0,line]:.3f} &{MSE_xp[1,line]:.3f} &{MSE_xp[2,line]:.3f} &{MSE_block[0,line]:.3f} &{MSE_block[1,line]:.3f} &{MSE_block[2,line]:.3f} &{MSE_did[line]:.3f}'+'\\'+'\\')

