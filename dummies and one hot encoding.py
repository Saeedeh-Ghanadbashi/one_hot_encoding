import pandas as pd
import numpy as np

db = pd.read_csv('./homeprices.csv')

dummies = pd.get_dummies(db)
dummies = dummies.drop(['town_west windsor'], axis = 'columns')

y = dummies.price
X = dummies.drop(['price'], axis = 'columns')

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X,y)

lr_model.score(X,y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dfle = db
db.town
dfle.town = le.fit_transform(dfle.town)

from sklearn.preprocessing import OneHotEncoder
X=dfle[['town', 'area']].values
y = dfle.price
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
X
ct.fit_transform(X)

X = X[:,1:]

lr_model.fit(X, y)
lr_model.score(X, y)