'''
Construct the domain matrix based on the historical price matrix. 
Could skip this step and use the original price matrix as is. 
Shibo Yao, May 8 2019
'''
import os
import numpy as np
import pandas as pd


df = pd.read_csv("sp500_price.csv", index_col = 0)
sp_info = pd.read_csv(os.path.abspath("../process/sp500_token.csv"))


start = '2016-08-24' # take a recent period for affinity matrix
end = '2018-08-27'
df = df.loc[start:end]
df = df.dropna(axis = 1, how = 'any') # drop those stocks contain null
df = df.pct_change(periods = 1) # convert price to return rate
df = df.dropna(axis = 0, how = 'any') # drop rows contain null
print(df.shape)
print(df.index[0], df.index[-1])

aff = df.corr() # can use different methods to define affinity

sectors = list(set(sp_info.Sector.values.tolist()))
dic_sec = {}
for i in range(len(sectors)):
    dic_sec[sectors[i]] = i+1

ind = aff.index.values # stock symbols full in blp format
y = []
Index = []
for i in ind:
    row = sp_info.Symbol.values.tolist().index(i.split(' ')[0])
    Index.append(sp_info.nGram.iloc[row])
    l = dic_sec[sp_info.Sector.iloc[row]]
    y.append(l)

aff.index = Index
y = pd.DataFrame({'y':y}, index = aff.index)
aff = aff.join(y)
print(aff.shape)

aff.to_csv("affMat.csv")
