# This function estimates ATT and it's corresponding bias and RMSE

import numpy as np
def ATT_estimate(Y_hat, Y, W):
    N,T = Y.shape
    treat_group = [i for i in range (N) if sum(W[i,:]) < T]
    ATT = np.zeros(len(treat_group))
    for i in range (len(treat_group)):
        unit = treat_group[i]
        ATT[i] = sum((Y[unit,:]-Y_hat[unit,:])*(1-W[unit,:])) / sum(1-W[unit,:])
    bias = np.abs(np.mean(ATT))
    RMSE = np.sqrt(np.mean(ATT**2))
    return ATT, bias, RMSE