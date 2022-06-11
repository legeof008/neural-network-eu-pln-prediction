
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


I will also use a variation of two kinds of inputs - 1 variable an 4 variable as shown below.
![alt text](plots/multivar2.svg?raw=true)
## The summary
As a naive method I "predicted" that the next day's price will be the same as the day before.
The LSTM network models seemed much more stable than the MLP ones.
In any other aspect though MLP prevails, it is faster to teach, takes less input and is less complicated and most importantly - it's more accurate.
The mean absolute errors of each solution are shown below.

MLP:
| Model        | 4 variables        | 1 variable         |
|--------------|--------------------|--------------------|
| 1            | 1.5430339340284637 | 1.4703023297557245 |
| 2            | 1.4492799562886722 | 1.4633906801606469 |
| 3            | 1.463395628647026  | 0.958279956288672  |
| naive method | 1.471203886470826  | 1.471203886470826  |

![alt text](plots/mlp_final.svg?raw=true)

LSTM:
| Model        | 4 variables        | 1 variable         |
|--------------|--------------------|--------------------|
| 1            | 1.5430339340284637 | 1.4703023297557245 |
| 2            | 1.4492799562886722 | 1.4633906801606469 |
| 3            | 1.463395628647026  | 0.958279956288672  |
| naive method | 1.471203886470826  | 1.471203886470826  |

![alt text](plots/lstm.svg?raw=true)

In summary I believe this kind of tool could be useful if paired with a fundamental analysis.
Though in some places the prediction seems to be changing trend too late which in real world could lead to selling or buying too late or too early, before the value peaks/hits bottom.
## Sources
https://www.tensorflow.org/api_docs  
https://www.researchgate.net/publication/\\335055073\_A\_Generative\_Neural\\\_Network\_Model\_for\_the\_Quality\_Prediction\_of\_Work\_in\_Progress\_Products  
https://arxiv.org/pdf/1502.06434.pdf  
https://www.heatonresearch.com/2017/06/01/hidden-layers.html  
https://expressanalytics.com/blog/neural-networks-prediction/
