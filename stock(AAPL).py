import pandas as pd
import matplotlib.pyplot as plt
import quandl
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'wPaa4PYkmNxA45kLj7f4'
stock_data = quandl.get('EOD/AAPL', start_date='2013-09-03', end_date='2017-12-28')
# print(stock_data)
dataset = pd.DataFrame(stock_data)
dataset.head()
dataset.to_csv('AAPL_stock.csv')

x = dataset.loc[:,'High':'Adj_Volume']
y = dataset.loc[:,'Open']

import sklearn.model_selection as model_selection
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size= 0.1,random_state = 0)
LR = LinearRegression()
LR.fit(x_train,y_train)
LR.score(x_test,y_test)

test_data = x.head(1)
prediction = LR.predict(test_data)
print('Predicted stock price:')
print(prediction)
print('Actual stock price:')
y.head(1)

