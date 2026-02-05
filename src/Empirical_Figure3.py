# This file generates Figure 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from pathlib import Path
from Estimator import Estimator, did_estimator

# Load the data
beer_sales = pd.read_csv('data/beer_sales.csv', index_col=[0])
treatment = pd.read_csv('data/treatment.csv', index_col=[0])
dates = beer_sales.columns.tolist()
date_format = "%Y-%m-%d"
dates = [datetime.strptime(str(i), date_format) for i in dates]

# Panel data Y and corresponding observation matrix
Y = beer_sales.to_numpy()
Y = Y
W = 1 - np.array(treatment)
N,T = Y.shape


# Estimation
y_hat_xp = Estimator(Y, W).PCA(4)   #PCA (k=4)
y_hat_causal = Estimator(Y, W).causalPCA_unknown_2(2)  #wi_PCA(k=2)
y_hat_block = Estimator(Y, W).block_PCA(4)  #Block-PCA(k=4)
y_hat_did = did_estimator(Y, W)  #TWFE


state = 'MA'
flag = beer_sales.index.get_loc(state)  #get the index of the row
treat_time = dates[np.sum(1-np.array(treatment.loc[state]))]

Y_obs = Y[flag, :]
Y_did = y_hat_did[flag, :]
Y_xp = y_hat_xp[flag, :]
Y_block = y_hat_block[flag, :]
Y_causal = y_hat_causal[flag, :]

# Plotting the curves
plt.figure(figsize=(9, 6))
plt.plot(dates, Y_obs, label='Observed', linestyle='--')
plt.plot(dates, Y_did, label='TWFE', color='red')
plt.plot(dates, Y_xp, label='PCA (k=4)', color='purple')
plt.plot(dates, Y_block, label='Block-PCA (k=4)', color='green')
plt.plot(dates, Y_causal, label='wi-PCA (k=2)')

handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[0], handles[4], handles[2], handles[3], handles[1]]
labels = [labels[0], labels[4], labels[2], labels[3], labels[1]]

# Adding vertical line
plt.axvline(x=treat_time, color='dimgrey', linestyle='dotted', linewidth=2.5)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Adding labels and title
plt.xlabel('Time', fontsize=11)
plt.ylabel('Beer Sales Per Store (in Dollars)', fontsize=11)
plt.legend(handles, labels, loc='upper left')
plt.grid()
plt.xlim(dates[0]-timedelta(days=6), dates[-1]+timedelta(days=6))
plt.savefig('./outputs/beer_'+state+'.png', dpi=500)
# plt.show()


state = 'NV'
flag = beer_sales.index.get_loc(state)  #get the index of the row
treat_time = dates[np.sum(1-np.array(treatment.loc[state]))]

Y_obs = Y[flag, :]
Y_did = y_hat_did[flag, :]
Y_xp = y_hat_xp[flag, :]
Y_block = y_hat_block[flag, :]
Y_causal = y_hat_causal[flag, :]

# Plotting the curves
plt.figure(figsize=(9, 6))
plt.plot(dates, Y_obs, label='Observed', linestyle='--')
plt.plot(dates, Y_did, label='TWFE', color='red')
plt.plot(dates, Y_xp, label='PCA (k=4)', color='purple')
plt.plot(dates, Y_block, label='Block-PCA (k=4)', color='green')
plt.plot(dates, Y_causal, label='wi-PCA (k=2)')

handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[0], handles[4], handles[2], handles[3], handles[1]]
labels = [labels[0], labels[4], labels[2], labels[3], labels[1]]

# Adding vertical line
plt.axvline(x=treat_time, color='dimgrey', linestyle='dotted', linewidth=2.5)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Adding labels and title
plt.xlabel('Time', fontsize=11)
plt.ylabel('Beer Sales Per Store (in Dollars)', fontsize=11)
plt.legend(handles, labels, loc='upper left')
plt.grid()
plt.xlim(dates[0]-timedelta(days=6), dates[-1]+timedelta(days=6))
plt.savefig('./outputs/beer_'+state+'.png', dpi=500)
# plt.show()
