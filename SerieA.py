import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

os.chdir('C:\\DATA SCIENCE\\Techolas Dataset\\Project\\Serie A')
match_df = pd.read_csv('sijil.csv')
match_df.head()

match_df.describe().T

match_df.info()

match_df.keys()

sns.countplot(x='FTR',data=match_df)

plt.figure(figsize=(20,10))
sns.heatmap(match_df.corr(),annot=True,cmap='Blues')

match_df['FTR'].value_counts()

match_df['AwayTeam'].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

match_df.HomeTeam = le.fit_transform(match_df.HomeTeam)
match_df.AwayTeam = le.fit_transform(match_df.AwayTeam)
match_df.FTR = le.fit_transform(match_df.FTR)

X = match_df.drop('FTR',axis=1)
y = match_df['FTR']

match_df['HomeTeam'].value_counts()

match_df.head()

from sklearn.model_selection import train_test_split

X_test, X_train, y_test, y_train = train_test_split(X,y, test_size=0.3, random_state=5)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model = LogisticRegression()
model.fit(X_train,y_train)

model.score(X_test, y_test)

Rfc = RandomForestClassifier(n_estimators=100)
Rfc.fit(X_train,y_train)

Rfc.score(X_test,y_test)

model.predict([[0,18,1,3,12,22,4,12,3,6,0,1]])

import pickle

pickle.dump(model, open('model_1.pkl','wb'))

