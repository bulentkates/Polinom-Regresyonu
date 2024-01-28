import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression  
dataset = pd.read_csv('dataset.csv')
dataset = dataset.dropna()
dataset = dataset[::-1].reset_index()
X = dataset.loc[:, ["Adj Close", "Close", "Low", "Open", "Volume"]].values
y = dataset.loc[:, ["High"]].values
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
X_pred_poly = poly_reg.transform(X)
y_pred = lin_reg.predict(X_pred_poly)
print(y_pred)
print(len(y_pred), "adet gün için tahmin yapılmıştır")
with open('tahminler.txt', 'w') as file:
    for prediction in y_pred:
        file.write(f"{prediction[0]}\n")