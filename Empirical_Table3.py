# This file generates Table 3

import numpy as np
import pandas as pd
from Estimator import Estimator, did_estimator
from ATT_estimate import ATT_estimate

# Load the data
beer_sales = pd.read_csv('./beer_sales.csv', index_col=[0])
treatment = pd.read_csv('./treatment.csv', index_col=[0])
dates = beer_sales.columns.tolist()

# Panel data Y and corresponding observation matrix
Y = beer_sales.to_numpy()
Y = Y / 100 # rescaled
W = 1 - np.array(treatment)
N,T = Y.shape


# take the control panel
control_group = [i for i in range (N) if sum(W[i,:]) == T]
Y = Y[control_group,:]

Simutime = 100
# k_set = np.arange(1,9).tolist()
k_set = np.arange(1,4).tolist()

# 1. uniformly random treatment pattern
print("--------------------------")
print("Uniformly random pattern")

def random_pattern(Y):
    N,T = Y.shape
    W = np.ones((N,T))
    choose_point = [t for t in range (T) if t%20 == 0]
    for t in choose_point:
        flag = np.random.binomial(n=1, p=0.8, size=N)
        W[:, t:t+20] = flag.reshape(-1,1)
    return W

np.random.seed(123)
did, xp, block, causal = np.zeros(3), np.zeros((len(k_set),3)), np.zeros((len(k_set),3)), np.zeros((len(k_set),3))
for s in range (Simutime):
    W = random_pattern(Y)   
    while (W @ W.T == 0).any():
        W = random_pattern(Y)

    # TWFE
    y_hat_did = did_estimator(Y, W)
    OOS = np.sqrt(np.sum(((Y-y_hat_did)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
    _, bias, RMSE = ATT_estimate(y_hat_did, Y, W)   # Bias_ATT, RMSE_ATT
    did += OOS, bias, RMSE

    for i in range (len(k_set)):  # test for different numbers of factors
        k = k_set[i]
        
        # PCA
        y_hat_xp = Estimator(Y, W).PCA(k)
        OOS = np.sqrt(np.sum(((Y-y_hat_xp)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_xp, Y, W)  # Bias_ATT, RMSE_ATT
        xp[i,:] += OOS, bias, RMSE
        
        # wi-PCA
        y_hat_causal = Estimator(Y, W).causalPCA_unknown_1(k)
        OOS = np.sqrt(np.sum(((Y-y_hat_causal)*(1-W))**2) / np.sum(1-W))  # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_causal, Y, W)  # Bias_ATT, RMSE_ATT
        causal[i,:] += OOS, bias, RMSE

did, xp, block, causal = did/Simutime, xp/Simutime, block/Simutime, causal/Simutime

print("--------------------------")
print("wi-PCA")
for k in k_set:
    print(f'&k={k} &{causal[k-1,0]:.3f} & {causal[k-1,1]:.3f} & {causal[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("PCA")
for k in k_set:
    print(f'&k={k} &{xp[k-1,0]:.3f} & {xp[k-1,1]:.3f} & {xp[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("block-PCA")
for k in k_set:
    print(f'&k={k} &- & - & -'+'\\'+'\\')
print("--------------------------")
print("TWFE")
print(f'& {did[0]:.3f} & {did[1]:.3f} & {did[2]:.3f}'+'\\'+'\\')



# 2. simultaneous treatment adoption
print("--------------------------")
print("Simultaneous treatment pattern")
def non_random_pattern_1(Y):
    N, T = Y.shape
    treat_group = []
    for i in range (N):
        tmp = np.random.binomial(n=1, p=0.5)
        if tmp == 1:
            treat_group.append(i)
    T0 = 140
    W = np.ones((N,T))
    W[treat_group, T0:] = 0
    return W

np.random.seed(123)
did, xp, block, causal = np.zeros(3), np.zeros((len(k_set),3)), np.zeros((len(k_set),3)), np.zeros((len(k_set),3))
for s in range (Simutime):
    W = non_random_pattern_1(Y) 

    # TWFE
    y_hat_did = did_estimator(Y, W)
    OOS = np.sqrt(np.sum(((Y-y_hat_did)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
    _, bias, RMSE = ATT_estimate(y_hat_did, Y, W)   # Bias_ATT, RMSE_ATT
    did += OOS, bias, RMSE

    for i in range (len(k_set)):  # test for different numbers of factors
        k = k_set[i]
        
        # PCA
        y_hat_xp = Estimator(Y, W).PCA(k)
        OOS = np.sqrt(np.sum(((Y-y_hat_xp)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_xp, Y, W)  # Bias_ATT, RMSE_ATT
        xp[i,:] += OOS, bias, RMSE

        # Block-PCA
        y_hat_block = Estimator(Y, W).block_PCA(k) 
        OOS = np.sqrt(np.sum(((Y-y_hat_block)*(1-W))**2) / np.sum(1-W))
        _, bias, RMSE = ATT_estimate(y_hat_block, Y, W)
        block[i,:] += float(OOS), bias, RMSE
        
        # wi-PCA
        y_hat_causal = Estimator(Y, W).causalPCA_unknown_2(k)   
        OOS = np.sqrt(np.sum(((Y-y_hat_causal)*(1-W))**2) / np.sum(1-W))  # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_causal, Y, W)  # Bias_ATT, RMSE_ATT
        causal[i,:] += OOS, bias, RMSE

did, xp, block, causal = did/Simutime, xp/Simutime, block/Simutime, causal/Simutime

print("--------------------------")
print("wi-PCA")
for k in k_set:
    print(f'&k={k} &{causal[k-1,0]:.3f} & {causal[k-1,1]:.3f} & {causal[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("PCA")
for k in k_set:
    print(f'&k={k} &{xp[k-1,0]:.3f} & {xp[k-1,1]:.3f} & {xp[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("block-PCA")
for k in k_set:
    print(f'&k={k} &- & - & -'+'\\'+'\\')
print("--------------------------")
print("TWFE")
print(f'& {did[0]:.3f} & {did[1]:.3f} & {did[2]:.3f}'+'\\'+'\\')



# 3. staggered treatment adoption
print("--------------------------")
print("Staggered treatment pattern")
# estimate fixed effects
def FE_PCA(Y):
    # Grand mean and fixed effects
    mu_hat = np.mean(Y)
    xi_hat = np.mean(Y, axis=0).reshape(-1,1) - mu_hat
    alpha_hat = np.mean(Y, axis=1).reshape(-1,1) - mu_hat 
    return xi_hat, alpha_hat

# staggered treatment adoption
def non_random_pattern_2(Y):
    xi_hat, alpha_hat = FE_PCA(Y)
    N = len(alpha_hat)
    T = len(xi_hat)
    W = np.ones((N,T))
    flag1 = [i for i in range (N) if alpha_hat[i] > 1]  
    flag2 = [t for t in range (T) if xi_hat[t] > 1]
    for i in flag1:
        tmp = np.random.binomial(n=1, p=0.9)
        if tmp == 1:
            for t in flag2:
                tmp = np.random.binomial(n=1, p=0.1)
                if tmp == 1:
                    W[i,t:] = 0
                    break
    return W


np.random.seed(123)
did, xp, block, causal = np.zeros(3), np.zeros((len(k_set),3)), np.zeros((len(k_set),3)), np.zeros((len(k_set),3))
for s in range (Simutime):
    W = non_random_pattern_2(Y) 

    # TWFE
    y_hat_did = did_estimator(Y, W)
    OOS = np.sqrt(np.sum(((Y-y_hat_did)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
    _, bias, RMSE = ATT_estimate(y_hat_did, Y, W)   # Bias_ATT, RMSE_ATT
    did += OOS, bias, RMSE

    for i in range (len(k_set)):  # test for different numbers of factors
        k = k_set[i]
        
        # PCA
        y_hat_xp = Estimator(Y, W).PCA(k)
        OOS = np.sqrt(np.sum(((Y-y_hat_xp)*(1-W))**2) / np.sum(1-W))   # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_xp, Y, W)  # Bias_ATT, RMSE_ATT
        xp[i,:] += OOS, bias, RMSE

        # Block-PCA
        y_hat_block = Estimator(Y, W).block_PCA(k) 
        OOS = np.sqrt(np.sum(((Y-y_hat_block)*(1-W))**2) / np.sum(1-W))
        _, bias, RMSE = ATT_estimate(y_hat_block, Y, W)
        block[i,:] += float(OOS), bias, RMSE
        
        # wi-PCA
        y_hat_causal = Estimator(Y, W).causalPCA_unknown_2(k)   
        OOS = np.sqrt(np.sum(((Y-y_hat_causal)*(1-W))**2) / np.sum(1-W))  # RMSE_Y
        _, bias, RMSE = ATT_estimate(y_hat_causal, Y, W)  # Bias_ATT, RMSE_ATT
        causal[i,:] += OOS, bias, RMSE

did, xp, block, causal = did/Simutime, xp/Simutime, block/Simutime, causal/Simutime

print("--------------------------")
print("wi-PCA")
for k in k_set:
    print(f'&k={k} &{causal[k-1,0]:.3f} & {causal[k-1,1]:.3f} & {causal[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("PCA")
for k in k_set:
    print(f'&k={k} &{xp[k-1,0]:.3f} & {xp[k-1,1]:.3f} & {xp[k-1,2]:.3f}'+'\\'+'\\')
print("--------------------------")
print("block-PCA")
for k in k_set:
    print(f'&k={k} &- & - & -'+'\\'+'\\')
print("--------------------------")
print("TWFE")
print(f'& {did[0]:.3f} & {did[1]:.3f} & {did[2]:.3f}'+'\\'+'\\')