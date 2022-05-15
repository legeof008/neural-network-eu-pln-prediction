import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten
from tensorflow.keras.models import Sequential

checked_currency = 'EUR'
against_currency = 'PLN'
predicion_days = 60


# Wczytywanie przebiegów
sheets = ['0.csv','1.csv','2.csv','3.csv']
data = []
for sheet in sheets:
    data+=pd.read_csv(sheet,usecols=['Close'])['Close'].to_list()

data = np.array(data)

# Przygotowanie danych

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data.reshape(-1,1))
x_train, y_train = [],[]

for i in range(0,4):
    x_train.append(scaled_data[i*predicion_days:(i*predicion_days)+predicion_days,0])
    y_train.append(scaled_data[(i*predicion_days)+predicion_days+1,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print(y_train)
print(len(x_train[0]))

# Budowa modelu

model = Sequential()
model.add(LSTM(units=40,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=40,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=40))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Przewidzenie następnego dnia

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=25,batch_size=32)

# Testowanie modelu

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

historical_data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', start, end)

test_start = dt.datetime(2020,2,2)+dt.timedelta(days=-predicion_days)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

test_dataset = pd.concat((historical_data['Close'],test_data['Close']),axis=0)

model_inputs = test_dataset[len(test_dataset) - len(test_data) - predicion_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test =[]

for x in range(predicion_days,len(model_inputs)):
    x_test.append(model_inputs[x-predicion_days:x,0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices,color ='black',label = 'Actual Prices')
plt.plot(prediction_prices,color='green',label='Predicted Prices')
plt.title(f'{checked_currency} price prediction')
plt.xlabel('Time [ days ]')
plt.ylabel('Price [ EUR/PLN ]')
plt.legend(loc='upper left')
plt.show()
