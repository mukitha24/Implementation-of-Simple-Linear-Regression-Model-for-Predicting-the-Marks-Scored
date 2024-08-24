# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Mukitha V M 
RegisterNumber:212223040119
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## Dataset
![DATASET 1](https://github.com/user-attachments/assets/8f44c647-e3b8-4190-9366-4edd29f82592)
## Head values
![head](https://github.com/user-attachments/assets/249f6689-3c9d-4750-a903-df29a7696079)
## Tail Values
![tail](https://github.com/user-attachments/assets/7e85cc27-3f23-441e-a712-a5ca11df4f10)
## X AND Y Values
![xyvalues](https://github.com/user-attachments/assets/dbcb8710-5f96-4f6b-b18f-09ac7c9a397b)
## Predication values of X and Y
![predict ](https://github.com/user-attachments/assets/667acc7c-0777-4b1b-88b0-34546319828a)
## MSE,MAE and RMSE
![values](https://github.com/user-attachments/assets/f133b06f-8f6d-4a20-af2b-031abd50b5cc)
## Training set
![trianing set](https://github.com/user-attachments/assets/72adf1eb-cf0b-43c8-bf21-ea662b221733)
## Testing set
![testing set](https://github.com/user-attachments/assets/f304d343-bc6b-4312-84a6-063e8e4e66d3)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
