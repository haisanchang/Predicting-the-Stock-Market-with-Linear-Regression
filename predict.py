import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#Read in the data set
df = pd.read_csv("sphist.csv")
df['Date'] = pd.to_datetime(df['Date'])

#Sort by the date column in ascending order
df = df.sort_values("Date", ascending=True)

#Add features that could be helpful for machine learning
df['5 Days Open'] = df['Open'].rolling(window=5).mean()
df['5 Days High'] = df['High'].rolling(window=5).mean()
df['5 Days Low'] = df['Low'].rolling(window=5).mean()
df['5 Days Volume'] = df['Volume'].rolling(window=5).mean()
df['Year'] = df['Date'].apply(lambda x: x.year)

#Adding Day of week column and set it to categorical
df['DOW'] = df['Date'].apply(lambda x: x.weekday())
dow_df = pd.get_dummies(df['DOW'])
df = pd.concat([df, dow_df], axis=1)
df = df.drop(['DOW'], axis=1)

#Because 'rolling' method above include the current date to compute, the current date include future knowledge, which means that the prediction will look not good in real world , so we shift all the values forward one day.
df['5 Days Open'] = df['5 Days Open'].shift(1)
df['5 Days High'] = df['5 Days High'].shift(1)
df['5 Days Low'] = df['5 Days Low'].shift(1)
df['5 Days Volume'] = df['5 Days Volume'].shift(1)

df = df[df['Date'] >= datetime(year=1951, month=1, day=3)]
df.dropna(axis=0)

#Split dataset into train set and test set.
train_df = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test_df = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

#Feature columns
train_colums = ['5 Days Open', '5 Days Volume', '5 Days High', '5 Days Low', 'Year', 0, 1, 2, 3, 4]

# Perform linear regression.
lr = LinearRegression()
lr.fit(train_df[train_colums], train_df['Close'])
prediction = lr.predict(test_df[train_colums])

test_df['prediction_close'] = prediction

test_df['difference'] = abs(test_df['Close'] - test_df['prediction_close'])

test_df['accuracy(%)'] = (1-(abs(test_df['Close'] - test_df['prediction_close']) / test_df['Close']
)) * 100
# Error metrics.
mse = mean_squared_error(test_df['Close'], prediction)
rmse = np.sqrt(mse)

print('mse:',mse)
print('rmse:',rmse)

print(test_df[['Date','Close','prediction_close','difference', 'accuracy(%)']].head(10))
