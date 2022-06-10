import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

checked_currency = 'EUR'
against_currency = 'PLN'

start = dt.datetime(2016,1,1)
end = dt.datetime(2021,1,1)

data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', start, end)

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data_0 = scaler.fit_transform(data['Close'].values.reshape(-1,1))

predicion_days = 5

x_train, y_train ,x_val, y_val = [],[],[],[]

for x in range(predicion_days,len(scaled_data_0)):
    if x % 4 != 0 :
        x_train.append(scaled_data_0[x-predicion_days:x,0])
        y_train.append(scaled_data_0[x,0])
    else :
        x_val.append(scaled_data_0[x-predicion_days:x,0])
        y_val.append(scaled_data_0[x,0])

x_train, y_train, x_val, y_val = np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1]))
x_val = np.reshape(x_val,(x_val.shape[0],x_val.shape[1]))
#print(y_train)


# Create the model
model = Sequential()
model.add(Dense(units=3,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(units=3))
model.add(Dense(units=1))

# Configure the model and start training
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=1500, batch_size=128, verbose=1, validation_split=0.2)

# Test dataset

test_start = dt.datetime(2020,2,2)+dt.timedelta(days=-predicion_days)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', test_start, test_end)

actual_prices = test_data['Close'].values

test_dataset_0 = pd.concat((data['Close'],test_data['Close']),axis=0)

model_inputs_0 = test_dataset_0[len(test_dataset_0) - len(test_data) - predicion_days:].values
model_inputs_0 = model_inputs_0.reshape(-1,1)
model_inputs_0 = scaler.fit_transform(model_inputs_0)


x_test =[]
y_test =[]


for x in range(predicion_days,len(model_inputs_0)):
    x_test.append(model_inputs_0[x-predicion_days:x,0])
    y_test.append(model_inputs_0[x,0])

x_test, y_test = np.array(x_test) , np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

##print(prediction_prices)

errs = []
naiv_err =[]
for x in range(0,len(actual_prices)-2):
    naiv_err.append(actual_prices[x+1])

# Plotting the output

plt.plot(actual_prices[0:len(actual_prices)-2],color ='black',label = 'Actual Prices')
plt.plot(prediction_prices[0:len(actual_prices)-2],color='green',label='Predicted Prices')
plt.plot(naiv_err,color='blue',label='Naive method')
plt.title(f'{checked_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
mape = mean_absolute_error(actual_prices[0:len(actual_prices)-2], prediction_prices[2:len(actual_prices)])*100
print(mape)
mape = mean_absolute_error(actual_prices[0:len(actual_prices)-2], naiv_err)*100
print(mape)