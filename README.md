# GA Capstone: Automated Stock Trading Agent
##### Will Arliss

---

#### Repository Contents

This repository is structured as follows:
 - sagsdg
   
---

## Problem

Can an intelligent stock trading agent be trained in a backtesting simulation environment to learn how to profitably trade stocks and automate the process of portfolio management? Reinforcement Learning AI techniques will be employed to train a Deep Q-Network on the historical prices of the **FANG** technology leaders. The goal of the model is not to predict prices, but rather to predict quantities of shares to trade such that a long term profit will be achieved.

## Data

Data used to populate the backtesting environment come from Yahoo Finance:
 - Facebook: [FB](https://finance.yahoo.com/quote/FB/history?period1=1337299200&period2=1601424000&interval=1d&filter=history&frequency=1d)
 - Apple: [AAPL](https://finance.yahoo.com/quote/AAPL/history?period1=345427200&period2=1601424000&interval=1d&filter=history&frequency=1d)
 - Netflix: [NFLX](https://finance.yahoo.com/quote/NFLX/history?period1=1022112000&period2=1601424000&interval=1d&filter=history&frequency=1d)
 - Google: [GOOG](https://finance.yahoo.com/quote/GOOG/history?period1=1092873600&period2=1601424000&interval=1d&filter=history&frequency=1d)
 
For simplicity, the environment will only observe prices starting in 2009. This will ensure that all data series are of the same length. It will also avoid the market shock of the 2008 Financial Crisis, a feature which could cause difficulty for training the model. Prices and volume will be fed to the agent normalized between 0 and 1 (as determined by the highest price/volumen yet seen in backtesting).

## Backtesting Environment

aasdg

## Modeling

agsds

## Next Steps

 - Could only build out on one stock at a time. In future, multiple stocks will be managed at once
 - Build a separate forecasting model to feed predictions to the agent 

---

#### Related Reading

 - djgal
