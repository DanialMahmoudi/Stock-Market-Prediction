#Importing The Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

#Importing The dataset & Training The Multiple Linear Regression & Evaluating The Model

dataset = pd.read_csv('Google_Stock_Price.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(len(y_train), 1))

regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train.flatten())

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))
np.set_printoptions(precision=2)
print('Real Values vs Predicted Values', end='\n\n')
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1), end='\n\n')

acc = r2_score(y_test, y_pred)
print("Polynomial Regression model's accuracy with degree equal to 2 is : ")
print(acc)

#Plotting The Real values & Predicted Ones

ls = []
for i in range(0, y_test.size):
  ls.append(i)
  pass
plt.plot(ls, y_test, c='red', label='Real Trend')
plt.plot(ls, y_pred, c='blue', label='Predicted Trend')
plt.legend()
plt.title('Real Trend vs Predicted Trend')
plt.xlabel('Day')
plt.ylabel('Open Value (y)')
plt.show()

#Predicting A New Dataset Using Previous Algorithm

dataset2 = pd.read_csv('Google_Stock_Price_Test.csv')
X2 = dataset2.iloc[:, :-1].values
y2 = dataset2.iloc[:, -1].values

y_pred2 = sc_y.inverse_transform(regressor.predict(sc_X.transform(X2)).reshape(-1, 1))
np.set_printoptions(precision=2)
print('Real Values vs Predicted Values', end='\n\n')
print(np.concatenate((y2.reshape(len(y2), 1), y_pred2.reshape(len(y_pred2), 1)), 1), end='\n\n')

acc = r2_score(y2, y_pred2)
print("Polynomial Regression model's accuracy with degree equal to 2 is : ")
print(acc)

#Plotting The Real values & Predicted Ones

ls = []
for i in range(0, y2.size):
  ls.append(i)
  pass
plt.plot(ls, y2, c='red', label='Real Trend')
plt.plot(ls, y_pred2, c='blue', label='Predicted Trend')
plt.legend()
plt.title('Real Trend vs Predicted Trend')
plt.xlabel('Day')
plt.ylabel('Open Value (y)')
plt.show()

