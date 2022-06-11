
# Neural networks currency prediction

This project aims to create a tool for easier technical analisys of stock related markets.



## Introduction to the problem
Currency exchange rate is an important analytical factor of estimating economic growth. As any other value exchange it is often a subject of speculation and for this reason keeping good information about certain currency values and where they're headed can be profitable.

An analysis of any value exchange comes down to two kinds of analysis :
- technical analysis,
- fundamental analysis.
The latter is about complete breakdown of certain ideas and looking deep within the clockwork of observed exchange rate. For stock exchange it might be for example an invastigation into the inner workings of a company etc. The former is a pure mathematical, methodical breakdown of the problem. That's the project's goal.

## Used frameworks
TensorFlow - an open source framework with a stable API.

## Network Model
I decided to use two examplary models for this project.
Classic MLP with BP learning:
```
model = Sequential()
model.add(Dense(units=3,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(units=3))
model.add(Dense(units=1))
```
LSTM model with dropout:
```
model.add(LSTM(units=3,return_sequences=True,activation='relu',input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=2,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=2))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```
![alt text](plots/zdj3.svg?raw=true)

## The input
The analysied exchange rate was **X=EUR/PLN**. The input is 3 chronological days, then the network produces the 4th day's price.
 
