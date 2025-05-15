import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("cars.csv")
print(df.head())

df.columns = pd.Index(df.columns).str.strip().str.lower()

df = df.drop(columns=["unnamed: 0"], errors='ignore')

X = df.drop(columns=["mpg"])
y = df["mpg"]

kf = KFold(n_splits=6, shuffle=True, random_state=42)

rootmeanse_list = []
r2_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rootmeanse_list.append(rmse)
    r2_list.append(r2)

print(f"Average RootMeanSE: {np.mean(rootmeanse_list):.4f}")
print(f"Average R²: {np.mean(r2_list):.4f}")
