import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

checked_currency = 'EUR'
against_currency = 'PLN'

starts = [dt.datetime(2015,8,1),dt.datetime(2016,4,1),dt.datetime(2016,12,1),dt.datetime(2020,1,1)]
ends = [dt.datetime(2015,8,1)+dt.timedelta(days=90),dt.datetime(2016,4,1)+dt.timedelta(days=90),dt.datetime(2016,12,1)+dt.timedelta(days=90),dt.datetime(2020,1,1)+dt.timedelta(days=90)]
for i in range(0,len(starts)):
    data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', starts[i], ends[i]).to_csv(str(i)+'.csv')
    

for i in range(0,4):
    data = pd.read_csv(str(i)+'.csv',usecols=['Close'])['Close'].to_list()
    plt.plot(data,color ='black',label = 'Actual Prices')
    plt.title(f'{checked_currency} price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()