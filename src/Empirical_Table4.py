# This file generates Table 4

import numpy as np
import pandas as pd
from scipy.stats import norm
from Estimator import Estimator, did_estimator

# Load the data
beer_sales = pd.read_csv('./beer_sales.csv', index_col=[0])
treatment = pd.read_csv('./treatment.csv', index_col=[0])
dates = beer_sales.columns.tolist()

# Panel data Y and corresponding observation matrix
Y = beer_sales.to_numpy()
Y = Y / 100 # rescaled
W = 1 - np.array(treatment)
N,T = Y.shape

alpha = 0.05
z = norm.ppf(1 - alpha/2)
B = 1000

treat_group = [i for i in range (N) if sum(W[i,:]) < T]
control_group = [i for i in range (N) if sum(W[i,:]) == T]
list = [0,4,1,2,5,3]   # according to the order in the paper
k_lst = [1,2,3,4]

print("--------------------------")
print("wi-PCA")
for k in k_lst:
    print(f"k={k}")
    np.random.seed(123)
    # Step 0: Estimate the common components
    C_hat = Estimator(Y, W).causalPCA_unknown_2(k) 

    SE_set = []
    ATT_hat_set = []
    for i in list:  # estimation for each state
        unit = treat_group[i]
        
        # Step 1: Point estimation
        ATT_hat = np.sum((Y[unit,:]-C_hat[unit,:])*(1-W[unit,:])) / np.sum(1-W[unit,:])
        
        # Step 2: Collect residual time series for SE
        eps_lst = []
        for j in control_group:
            W_j = W.copy()
            W_j[j,:] = W[unit,:]
            C_hat_j = Estimator(Y, W_j).causalPCA_unknown_2(k)
            eps = Y[j,:] - C_hat_j[j,:]
            eps_lst.append(eps)

        # Step 3: Resampling bootstrap
        ATT_hat_bs = np.zeros((B))
        for b in range (B):
            sample_error = np.random.randint(0, len(control_group))
            Y_bs_unit = C_hat[unit,:].reshape(1,-1) + eps_lst[sample_error].reshape(1,T)
            sample = np.random.choice([i for i in range (N) if i != unit], size=N-1, replace=True)
            Y_bs = np.concatenate((Y[sample,:], Y_bs_unit), axis=0)
            W_bs = np.concatenate((W[sample,:], W[unit,:].reshape(1,-1)), axis=0)
            C_hat_bs = Estimator(Y_bs, W_bs).causalPCA_unknown_2(k)
            ATT_hat_bs[b] = np.sum((Y_bs[-1,:]-C_hat_bs[-1,:])*(1-W_bs[-1,:])) / np.sum(1-W_bs[-1,:])
        V_hat = np.var(ATT_hat_bs)  
        ATT_hat_set.append(ATT_hat)
        SE_set.append(np.sqrt(V_hat))
        # print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f} &({(ATT_hat-z*np.sqrt(V_hat)):.2f}, {(ATT_hat+z*np.sqrt(V_hat)):.2f})') 
        print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f}') 


print("--------------------------")
print("PCA")
for k in k_lst:
    print(f"k={k}")
    np.random.seed(123)
    # Step 0: Estimate the common components
    C_hat = Estimator(Y, W).PCA(k)    #did_estimator(Y, W)            

    SE_set = []
    ATT_hat_set = []
    for i in list:  # estimation for each state
        unit = treat_group[i]
        
        # Step 1: Point estimation
        ATT_hat = np.sum((Y[unit,:]-C_hat[unit,:])*(1-W[unit,:])) / np.sum(1-W[unit,:])
        
        # Step 2: Collect residual time series for SE
        eps_lst = []
        for j in control_group:
            W_j = W.copy()
            W_j[j,:] = W[unit,:]
            C_hat_j = Estimator(Y, W_j).PCA(k)
            eps = Y[j,:] - C_hat_j[j,:]
            eps_lst.append(eps)

        # Step 3: Resampling bootstrap
        ATT_hat_bs = np.zeros((B))
        for b in range (B):
            sample_error = np.random.randint(0, len(control_group))
            Y_bs_unit = C_hat[unit,:].reshape(1,-1) + eps_lst[sample_error].reshape(1,T)
            sample = np.random.choice([i for i in range (N) if i != unit], size=N-1, replace=True)
            Y_bs = np.concatenate((Y[sample,:], Y_bs_unit), axis=0)
            W_bs = np.concatenate((W[sample,:], W[unit,:].reshape(1,-1)), axis=0)
            C_hat_bs = Estimator(Y_bs, W_bs).PCA(k)            
            ATT_hat_bs[b] = np.sum((Y_bs[-1,:]-C_hat_bs[-1,:])*(1-W_bs[-1,:])) / np.sum(1-W_bs[-1,:])
        V_hat = np.var(ATT_hat_bs)  
        ATT_hat_set.append(ATT_hat)
        SE_set.append(np.sqrt(V_hat))
        # print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f} &({(ATT_hat-z*np.sqrt(V_hat)):.2f}, {(ATT_hat+z*np.sqrt(V_hat)):.2f})') 
        print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f}') 


print("--------------------------")
print("block-PCA")
for k in k_lst:
    print(f"k={k}")
    np.random.seed(123)
    # Step 0: Estimate the common components
    C_hat = Estimator(Y, W).block_PCA(k)    #did_estimator(Y, W)            

    SE_set = []
    ATT_hat_set = []
    for i in list:  # estimation for each state
        unit = treat_group[i]
        
        # Step 1: Point estimation
        ATT_hat = np.sum((Y[unit,:]-C_hat[unit,:])*(1-W[unit,:])) / np.sum(1-W[unit,:])
        
        # Step 2: Collect residual time series for SE
        eps_lst = []
        for j in control_group:
            W_j = W.copy()
            W_j[j,:] = W[unit,:]
            C_hat_j = Estimator(Y, W_j).block_PCA(k)
            eps = Y[j,:] - C_hat_j[j,:]
            eps_lst.append(eps)

        # Step 3: Resampling bootstrap
        ATT_hat_bs = np.zeros((B))
        for b in range (B):
            sample_error = np.random.randint(0, len(control_group))
            Y_bs_unit = C_hat[unit,:].reshape(1,-1) + eps_lst[sample_error].reshape(1,T)
            sample = np.random.choice([i for i in range (N) if i != unit], size=N-1, replace=True)
            Y_bs = np.concatenate((Y[sample,:], Y_bs_unit), axis=0)
            W_bs = np.concatenate((W[sample,:], W[unit,:].reshape(1,-1)), axis=0)
            C_hat_bs = Estimator(Y_bs, W_bs).block_PCA(k) 
            ATT_hat_bs[b] = np.sum((Y_bs[-1,:]-C_hat_bs[-1,:])*(1-W_bs[-1,:])) / np.sum(1-W_bs[-1,:])
        V_hat = np.var(ATT_hat_bs)  
        ATT_hat_set.append(ATT_hat)
        SE_set.append(np.sqrt(V_hat))
        # print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f} &({(ATT_hat-z*np.sqrt(V_hat)):.2f}, {(ATT_hat+z*np.sqrt(V_hat)):.2f})') 
        print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f}') 


print("--------------------------")
print("TWFE")
# Step 0: Estimate the common components
C_hat = did_estimator(Y, W)            

SE_set = []
ATT_hat_set = []
for i in list:  # estimation for each state
    unit = treat_group[i]
    
    # Step 1: Point estimation
    ATT_hat = np.sum((Y[unit,:]-C_hat[unit,:])*(1-W[unit,:])) / np.sum(1-W[unit,:])
    
    # Step 2: Collect residual time series for SE
    eps_lst = []
    for j in control_group:
        W_j = W.copy()
        W_j[j,:] = W[unit,:]
        C_hat_j = did_estimator(Y, W_j)  # can use Estimator(Y, W_j).causalPCA_unknown_2(0) to approximate
        eps = Y[j,:] - C_hat_j[j,:]
        eps_lst.append(eps)

    # Step 3: Resampling bootstrap
    ATT_hat_bs = np.zeros((B))
    for b in range (B):
        sample_error = np.random.randint(0, len(control_group))
        Y_bs_unit = C_hat[unit,:].reshape(1,-1) + eps_lst[sample_error].reshape(1,T)
        sample = np.random.choice([i for i in range (N) if i != unit], size=N-1, replace=True)
        Y_bs = np.concatenate((Y[sample,:], Y_bs_unit), axis=0)
        W_bs = np.concatenate((W[sample,:], W[unit,:].reshape(1,-1)), axis=0)
        C_hat_bs = did_estimator(Y_bs, W_bs)    # can use Estimator(Y_bs, W_bs).causalPCA_unknown_2(0) to approximate        
        ATT_hat_bs[b] = np.sum((Y_bs[-1,:]-C_hat_bs[-1,:])*(1-W_bs[-1,:])) / np.sum(1-W_bs[-1,:])
    V_hat = np.var(ATT_hat_bs)  
    ATT_hat_set.append(ATT_hat)
    SE_set.append(np.sqrt(V_hat))
    # print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f} &({(ATT_hat-z*np.sqrt(V_hat)):.2f}, {(ATT_hat+z*np.sqrt(V_hat)):.2f})') 
    print(f'&{ATT_hat:.2f} &{np.sqrt(V_hat):.2f}') 