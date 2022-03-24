import numpy as np
import pandas as pd
import pandas_datareader as data
from keras.models import load_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')
import streamlit as st

start='2010-01-01'
end='2019-12-31'
st.title('Stock Price Prediction')
user_input = st.text_input('Enter Stock Ticker : ','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart ')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#Create a variable to predict 'x' days out into the future
future_days = 25
#create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)
df.tail(4)

X = np.array(df.drop(['Prediction'], 1))[:-future_days]
#Create the target data set (y) and convert it to a numpy array and get all of thev target values except the last 'x' rows/days
y = np.array(df['Prediction'])[:-future_days]

#Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Create the models
#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)


#Get the last 'x' rows of the feature data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_Prediction = tree.predict(x_future)
print(tree_Prediction)
print()
#Show the model linear regression Prediction
lr_Prediction = lr.predict(x_future)

#Visualize the data
Predictions = tree_Prediction

valid = df[X.shape[0]:]
valid['Predictions'] = Predictions
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(10,5))
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','val','Pred'])
st.pyplot(fig2)


