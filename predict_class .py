import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

df = pd.read_csv('teleCust1000t.csv')
df.head()

df['custcat'].value_counts()

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]
y = df['custcat'].values
y[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh
yhat = neigh.predict(X_test)
yhat[0:5]
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


#Sample Data Set

 #	region 	tenure 	age 	marital address income 	ed 	employ 	retire 	gender 	reside 	custcat
#0 	2 	13 	44 	1 	9 	64.0 	4 	5 	0.0 	0 	2 	1
#1 	3 	11 	33 	1 	7 	136.0 	5 	5 	0.0 	0 	6 	4   --> This all are classes.
#2 	3 	68 	52 	1 	24 	116.0 	1 	29 	0.0 	1 	2 	3
#3 	2 	33 	33 	0 	12 	33.0 	2 	0 	0.0 	1 	1 	1
#4 	2 	23 	30 	1 	9 	30.0 	1 	2 	0.0 	0 	4 	3
