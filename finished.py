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
end = dt.datetime(2020,2,2)

data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', start, end)

# Przygotowywanie danych 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_0 = scaler.fit_transform(data['Close'].values.reshape(-1,1))
scaled_data_1 = scaler.fit_transform(data['Open'].values.reshape(-1,1))
scaled_data_2 = scaler.fit_transform(data['Volume'].values.reshape(-1,1))
scaled_data_3 = scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))


predicion_days = 60

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
print(len(x_train))
print(len(x_val))

# Model sieci neuronowej

# Model 1
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,activation='relu',input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Przewidzenie następnego dnia

# Model 2
#model = Sequential()
#
#model.add(LSTM(units=30,return_sequences=True,input_shape=(x_train.shape[1],1)))
#model.add(Dropout(0.2))
#model.add(LSTM(units=30,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units=30))
#model.add(Dropout(0.2))
#model.add(Dense(units=1)) # Przewidzenie następnego dnia

# Model 3
#model = Sequential()
#
#model.add(LSTM(units=20,return_sequences=True,input_shape=(x_train.shape[1],1)))
#model.add(Dropout(0.2))
#model.add(LSTM(units=20,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units=20))
#model.add(Dropout(0.2))
#model.add(Dense(units=1)) # Przewidzenie następnego dnia



model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=1500,batch_size=128,validation_data=(x_val,y_val))

# Testowanie modelu

test_start = dt.datetime(2020,2,2)+dt.timedelta(days=-predicion_days)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{checked_currency}{against_currency}=X', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

test_dataset_0 = pd.concat((data['Close'],test_data['Close']),axis=0)
test_dataset_1 = pd.concat((data['Open'],test_data['Open']),axis=0)
test_dataset_2 = pd.concat((data['Volume'],test_data['Volume']),axis=0)
test_dataset_3 = pd.concat((data['Adj Close'],test_data['Adj Close']),axis=0)

model_inputs_0 = test_dataset_0[len(test_dataset_0) - len(test_data) - predicion_days:].values
model_inputs_0 = model_inputs_0.reshape(-1,1)
model_inputs_0 = scaler.fit_transform(model_inputs_0)

model_inputs_1 = test_dataset_1[len(test_dataset_1) - len(test_data) - predicion_days:].values
model_inputs_1 = model_inputs_1.reshape(-1,1)
model_inputs_1 = scaler.fit_transform(model_inputs_1)

model_inputs_2 = test_dataset_2[len(test_dataset_2) - len(test_data) - predicion_days:].values
model_inputs_2 = model_inputs_2.reshape(-1,1)
model_inputs_2 = scaler.fit_transform(model_inputs_2)

model_inputs_3 = test_dataset_3[len(test_dataset_3) - len(test_data) - predicion_days:].values
model_inputs_3 = model_inputs_3.reshape(-1,1)
model_inputs_3 = scaler.fit_transform(model_inputs_3)

x_test =[]
y_test =[]

for x in range(predicion_days,len(model_inputs_0)):
    x_test.append(model_inputs_0[x-predicion_days:x,0])
    y_test.append(model_inputs_0[x,0])

x_test, y_test = np.array(x_test) , np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

errs = []
naiv_err =[]
for x in range(0,len(actual_prices)-1):
    naiv_err.append(actual_prices[x+1])

plt.plot(actual_prices[0:len(actual_prices)-1],color ='black',label = 'Actual Prices')
plt.plot(prediction_prices[0:len(actual_prices)-1],color='green',label='Predicted Prices')
plt.plot(naiv_err,color='blue',label='Naive method')
plt.title(f'{checked_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()



plt.boxplot(errs)
plt.boxplot(naiv_err)
plt.title('Absolute error of method')
plt.xlabel('Prediction number')
plt.ylabel('Error')
plt.legend(loc='upper left')
#plt.show()


_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plt.show()

print(len(x_train))
print(len(x_val))
print(len(x_test))
mape = mean_absolute_error(actual_prices, prediction_prices)*100
print(mape)
mape = mean_absolute_error(actual_prices[0:len(actual_prices)-1], naiv_err)*100
print(mape)
# Rzeczywiste przewidywania
#
#real_data = [model_inputs[len(model_inputs) - predicion_days:len(model_inputs) ,0]]
#real_data = np.array(real_data)
#real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))
#
#prediction = model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
#