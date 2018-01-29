import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
df=df[['Open','Close']]
df=df[:]
print(df)
