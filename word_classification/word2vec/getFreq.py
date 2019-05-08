'''
Get word frequencies. 
Shibo Yao, May 8 2019
'''
import os
import collections
import pandas as pd


sp_info = pd.read_csv(os.path.abspath("../process/sp500_token.csv"))
stocks = sp_info.nGram
stocks = [s.lower() for s in stocks]

with open (os.path.abspath("../process/processed.txt")) as f:
    words = f.read().split(' ')

count = collections.Counter(words)

key = []
freq = []
for s in stocks:
    try:
        freq.append(count[s])
        key.append(s)
    except:
        pass

df = pd.DataFrame({'Token':key, 'Freq':freq})
df.to_csv("tokenFreq.csv")
