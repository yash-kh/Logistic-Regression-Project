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
![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

# Exploratory Data Analysis
## Missing Data
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this or change it to another feature like "Cabin Known: 1 or 0"

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
```
![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)
```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
```
![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)

![image](https://www.ex-t.com/wp-content/uploads/2019/04/blank-160.png)
