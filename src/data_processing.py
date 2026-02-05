# This file organizes data from the original source

import numpy as np
import pandas as pd
from datetime import datetime

years = [str(t) for t in range (2011, 2021)]
products = ['5000', '5001', '5005', '5010', '5015', '5020']
dir = './data/'

# organize the data
for year in years:
    # store panel
    df_store = pd.read_csv(dir + 'nielsen_extracts/RMS/'+year+'/Annual_Files/stores_'+year+'.tsv', sep='\t')
    df_store = df_store[['store_code_uc','fips_state_descr']]

    # product panel    
    for product in products:
        df = pd.read_csv(dir + 'nielsen_extracts/RMS/'+year+'/Movement_Files/5001_'+year+'/'+product+'_'+year+'.tsv', sep='\t')
        df['dollar'] = df['price'] / df['prmult'] * df['units']
        df = df[['store_code_uc','week_end','dollar']]
        df = pd.merge(df, df_store, on='store_code_uc', how='left')

        grouped_df = df.groupby(['fips_state_descr','week_end'])
        df1 = grouped_df['dollar'].sum()
        df2 = grouped_df['store_code_uc'].nunique()
        df = pd.merge(df1, df2, left_index=True, right_index=True)
        df = df.rename(columns={'store_code_uc':'store'})

        df.to_csv(dir + 'panels/details/'+year+'_'+product+'.csv', index=True)

for year in years:
    df = pd.read_csv(dir + 'panels/details/'+year+'_5000.csv', index_col=[0,1])
    for product in products[1:]:
        df_tmp = pd.read_csv(dir + 'panels/details/'+year+'_'+product+'.csv', index_col=[0,1])
        df = df.add(df_tmp, fill_value=0)
    df['store'] = df['store'].astype('int')

    # get 48 states
    if 'AK' in df.index.get_level_values(0):
        df.drop('AK', level=0, inplace=True)
    if 'HI' in df.index.get_level_values(0):
        df.drop('HI', level=0, inplace=True)
    if 'DC' in df.index.get_level_values(0):
        df.drop('DC', level=0, inplace=True)

    df.to_csv(dir + 'panels/beer_'+year+'.csv', index=True)


# treatment time
treatment = {'NV': 20170701, 'CA': 20180101, 'CO': 20140101, 'MI': 20191201, 'IL': 20200101, 'OR': 20151001, 'WA': 20140708, 'ME': 20201009, 'MA': 20181120}


# construct time series data (beer sales and treatment) for each state
df = pd.read_csv(dir + 'panels/beer_2011.csv', index_col=[0,1])
years = [str(t) for t in range (2011, 2021)]
states = list(set(df.index.get_level_values(0)))

for state in states:
    df = pd.DataFrame({})
    for year in years:
        df_tmp = pd.read_csv(dir + 'panels/beer_'+year+'.csv', index_col=[0,1])
        df_tmp = df_tmp.loc[state]
        df = pd.concat([df, df_tmp])
    df['treat'] = 0
    if state in treatment:
        df.loc[df.index > treatment[state], 'treat'] = 1

    df.to_csv(dir + 'panels/states/beer_'+state+'.csv', index=True)


# get states name
dir = './empirics/'
df = pd.read_csv(dir + 'panels/beer_2011.csv', index_col=[0,1])
states = list(set(df.index.get_level_values(0)))
drop_states = ['CO', 'WA', 'OR']
states = [s for s in states if s not in drop_states]
states.sort()

# construct panel data starting from 2017
start = 20170101
sales, stores, treats = [], [], []
for state in states:
    df = pd.read_csv(dir + 'panels/states/beer_'+state+'.csv', index_col=0)
    sales.append(list(df.loc[df.index >=start, 'dollar'].values))
    stores.append(list(df.loc[df.index >=start,'store'].values))
    treats.append(list(df.loc[df.index >=start,'treat'].values))

Y = np.array(sales) / np.array(stores)
date_format = "%Y%m%d"
dates = list(df.loc[df.index >=start].index.values)
dates = [datetime.strptime(str(i), date_format) for i in dates]
data_Y = pd.DataFrame(Y, columns=dates, index=states)
data_W = pd.DataFrame(treats, columns=dates, index=states)
data_Y.to_csv(dir + 'beer_sales.csv')   # beer sales per store
data_W.to_csv(dir + 'treatment.csv')   # treatment


# generate contaminated beer sales data
noise = np.random.normal(loc=0.0, scale=100, size=data_Y.shape)
beer_sales_contaminated = data_Y + noise
beer_sales_contaminated.to_csv(dir + 'beer_sales_contaminated.csv')