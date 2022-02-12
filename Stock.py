#Install the dependencies 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#Load the data 
from google.colab import files
uploaded = files.upload()

#Store the data into a data frame
df = pd.read_csv('TSLA.csv')
df.head(6)

#Get the number of trading days
df.shape

#Visualize the close price data
plt.figure(figsize=(16,8))
plt.title('tesla')
plt.xlabel('Days')
plt.ylabel('Close USD($)')
plt.plot(df['Close'])
plt.show()

#Get the close price
df = df[['Close']]
df.head(4)

#Create a variable to predict 'x' days out into the future
future_days = 25
#create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)
df.tail(4)

#create the feature data set (X) and convert it to a numpy array and remove the last 'X' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)

#Create the target data set (y) and convert it to a numpy array and get all of thev target values except the last 'x' rows/days
y = np.array(df['Prediction'])[:-future_days]
print(y)

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
x_future

#Show the model Decision tree Prediction
tree_Prediction = tree.predict(x_future)
print(tree_Prediction)
print()
#Show the model linear regression Prediction
lr_Prediction = lr.predict(x_future)
print(lr_Prediction)

#Visualize the data
Predictions = tree_Prediction

valid = df[X.shape[0]:]
valid['Predictions'] = Predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','val','Pred'])
plt.show()

