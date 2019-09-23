# -*- coding: utf-8 -*-
"""


@author: ravi teja
"""
## Problem statment 
## We will working with a fake advertisment dataset, indicating whether or not a 
##particular internet user click on an advertisment. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline

## Get the Data - Read in the advertising.csv file and set it to a data frame called ad_data.**
ad_data = pd.read_csv("C:\\Users\\ravi\\Desktop\\INDRAS ACADEMY\\phyton programming\\govardhan phyton\\Python_P3\\Machine Learning\\Logistic\\advertising.csv")
ad_data.head()

## Use info and describe() on ad_data
ad_data.info()
ad_data.describe() 

###Exploratory Data Analysis
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

###Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age',y='Area Income',data=ad_data)

#### Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');

## Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

###create a pairplot with the hue defined by the 'Clicked on Ad' column feature
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

###train test split, and train our model

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

## Predictions and Evaluations
predictions = logmodel.predict(X_test)
predictions


# Create a classification report for the model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))




