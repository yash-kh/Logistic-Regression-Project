# Logistic Regression Project

In this project we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic).

We'll be trying to predict a classification- survival or deceased.

## Import Libraries
Let's import some libraries to get started!

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
# The Data
```python
train = pd.read_csv('titanic_train.csv')
train.head()
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/1.PNG?raw=true)

# Exploratory Data Analysis
## Missing Data
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/2.png?raw=true)

Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this or change it to another feature like "Cabin Known: 1 or 0"

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
```

![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/3.png?raw=true)

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
```

![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/4.png?raw=true)

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
```

![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/5.png?raw=true)

```python
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
```

![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/6.png?raw=true)

```python
sns.countplot(x='SibSp',data=train)
```

![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/7.png?raw=true)

```python
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/8.png?raw=true)

## Data Cleaning
We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
However we can be smarter about this and check the average age by passenger class. For example:
```python
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)
```
We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.
```python
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
```
```python
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
```
Now let's check that heat map again!
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/9.png?raw=true)

Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.
```python
train.drop('Cabin',axis=1,inplace=True)
```
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/10.png?raw=true)
```python
train = train.dropna()
```
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/11.png?raw=true)

# Converting Categorical Features 

We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

```python
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
```
```python
train.head()
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/12.PNG?raw=true)

# Building a Logistic Regression model

Let's start by splitting our data into a training set and test set

## Train Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
```
## Training and Predicting
```python
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
```
## Evaluation
```python
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```
![image](https://github.com/yash-kh/Logistic-Regression-Project/blob/master/plots/13.PNG?raw=true)

At last we got a score of 82% accuracy on this model.
