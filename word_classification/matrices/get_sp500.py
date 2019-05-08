'''
Grab historical price data of SP500 companies from Bloomberg terminal.
Shibo Yao, May 8 2019
'''
import os
import numpy as np
import pandas as pd
import tia.bbg.datamgr as dm


csv = pd.read_csv(os.path.abspath("../process/sp500_token.csv"))
stocks = csv.Symbol.values.tolist()
stocks = [s+' US EQUITY' for s in stocks]


mgr = dm.BbgDataManager()
seeds = mgr[stocks]
df = seeds.get_historical('PX_LAST', '10/1/2008', '10/1/2018')


df.to_csv("sp500_price.csv")
