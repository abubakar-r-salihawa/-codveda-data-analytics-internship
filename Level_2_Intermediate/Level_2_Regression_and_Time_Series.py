#!/usr/bin/env python
# coding: utf-8

# # Codveda Data Analytics Internship - Level 2 (Intermediate)
#
# **Intern:** Abubakar Rabiu Salihawa
# **Internship:** Data Analysis Intern, Codveda Technologies
# **Intern ID:** CV/AI/82271
#
# Selected tasks: Task 1 - Regression Analysis and Task 2 - Time Series Analysis.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10
})

ROOT = Path.cwd().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs' / 'level2'
OUT.mkdir(parents=True, exist_ok=True)

# Task 1: Regression Analysis
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
housing = pd.read_csv(DATA / 'house_prediction.txt', sep=r'\s+', header=None, names=columns)
print('Dataset shape:', housing.shape)
print(housing.head())
print(housing[['RM','MEDV']].describe().round(3))

X = housing[['RM']]
y = housing['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
coefficient = model.coef_[0]
intercept = model.intercept_

metrics = pd.DataFrame({
    'Metric': ['Intercept', 'Coefficient for RM', 'R-squared', 'Mean Squared Error', 'Root Mean Squared Error'],
    'Value': [intercept, coefficient, r2, mse, rmse]
})
metrics.to_csv(OUT / 'regression_metrics.csv', index=False)
print(metrics.round(4))
print(f'Estimated equation: MEDV = {intercept:.3f} + ({coefficient:.3f} × RM)')
print(f'Interpretation: one additional room is associated with an estimated ${coefficient*1000:,.0f} increase in median house value, on average.')

plt.figure(figsize=(9, 6))
plt.scatter(X_test['RM'], y_test, alpha=0.75, label='Actual test observations')
order = np.argsort(X_test['RM'].values)
plt.plot(X_test['RM'].values[order], y_pred[order], linewidth=2.5, label='Regression line')
plt.title('Simple Linear Regression: Rooms and House Value')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median House Value, $000s (MEDV)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'regression_rooms_vs_value.png', dpi=200, bbox_inches='tight')
plt.close()

# Task 2: Time Series Analysis
stock = pd.read_csv(DATA / 'stock_prices.csv', parse_dates=['date'])
aapl = (stock.loc[stock['symbol'].eq('AAPL'), ['date','close']]
        .dropna()
        .drop_duplicates('date')
        .sort_values('date')
        .set_index('date'))

business_index = pd.date_range(aapl.index.min(), aapl.index.max(), freq='B')
aapl_b = aapl.reindex(business_index)
aapl_b.index.name = 'date'
aapl_b['close'] = aapl_b['close'].interpolate(method='time').ffill().bfill()
aapl_b['MA_20'] = aapl_b['close'].rolling(20).mean()
aapl_b['MA_60'] = aapl_b['close'].rolling(60).mean()

print('Date range:', aapl_b.index.min().date(), 'to', aapl_b.index.max().date())
print('Business-day observations:', len(aapl_b))
print(aapl_b.head())

plt.figure(figsize=(13, 6))
plt.plot(aapl_b.index, aapl_b['close'], linewidth=1, label='Daily close')
plt.plot(aapl_b.index, aapl_b['MA_20'], linewidth=1.8, label='20-day moving average')
plt.plot(aapl_b.index, aapl_b['MA_60'], linewidth=2.2, label='60-day moving average')
plt.title('AAPL Closing Price with Moving-Average Smoothing')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'aapl_moving_averages.png', dpi=200, bbox_inches='tight')
plt.close()

# 21 business days approximates one trading month.
decomposition = seasonal_decompose(aapl_b['close'], model='additive', period=21, extrapolate_trend='freq')
fig = decomposition.plot()
fig.set_size_inches(13, 10)
fig.suptitle('AAPL Additive Time-Series Decomposition (21 Business Days)', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'aapl_decomposition.png', dpi=200, bbox_inches='tight')
plt.close()

components = pd.DataFrame({
    'close': aapl_b['close'],
    'trend': decomposition.trend,
    'seasonal': decomposition.seasonal,
    'residual': decomposition.resid,
    'MA_20': aapl_b['MA_20'],
    'MA_60': aapl_b['MA_60']
})
components.to_csv(OUT / 'aapl_time_series_components.csv')
print(components.dropna().head())

print('\nLevel 2 complete: regression and time-series outputs saved to outputs/level2.')
