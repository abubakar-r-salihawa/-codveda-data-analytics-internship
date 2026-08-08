#!/usr/bin/env python
# coding: utf-8

# # Codveda Data Analytics Internship - Level 1 (Basic)
# 
# **Intern:** Abubakar Rabiu Salihawa  
# **Internship:** Data Analysis Intern, Codveda Technologies  
# **Intern ID:** CV/AI/82271  
# 
# Selected tasks: **Task 1 - Data Cleaning and Preprocessing** and **Task 2 - Exploratory Data Analysis (EDA)**.
# 
# This notebook records the steps I followed, the results obtained, and what the results mean. The code is included so that the analysis can be checked and repeated.

# ## Task 1: Data Cleaning and Preprocessing
# 
# The stock-price dataset is used because it contains missing values in the `open`, `high`, and `low` columns, a date column that must be converted to a proper datetime type, and many stock symbols that should be standardized. The workflow also checks and removes duplicate rows even when none are found.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10
})
import seaborn as sns

ROOT = Path.cwd().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs' / 'level1'
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 20)

stock_raw = pd.read_csv(DATA / 'stock_prices.csv')
print('Original shape:', stock_raw.shape)
print('Duplicate rows:', stock_raw.duplicated().sum())
print('Missing values before cleaning:')
print(stock_raw.isna().sum().to_frame('missing_count'))
print(stock_raw.head())

stock = stock_raw.copy()
stock.columns = stock.columns.str.strip().str.lower().str.replace(' ', '_')
stock['symbol'] = stock['symbol'].astype(str).str.strip().str.upper()
stock['date'] = pd.to_datetime(stock['date'], errors='coerce')

rows_before = len(stock)
stock = stock.drop_duplicates().copy()
duplicates_removed = rows_before - len(stock)
stock = stock.sort_values(['symbol', 'date']).reset_index(drop=True)

price_cols = ['open', 'high', 'low', 'close']
stock[price_cols] = stock.groupby('symbol')[price_cols].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
for col in price_cols:
    stock[col] = stock[col].fillna(stock.groupby('symbol')[col].transform('median'))
    stock[col] = stock[col].fillna(stock[col].median())

stock = stock.dropna(subset=['symbol', 'date', 'close']).reset_index(drop=True)
stock['high'] = stock[['open', 'high', 'low', 'close']].max(axis=1)
stock['low'] = stock[['open', 'high', 'low', 'close']].min(axis=1)
stock['volume'] = pd.to_numeric(stock['volume'], errors='coerce').fillna(0).clip(lower=0).astype('int64')
stock.to_csv(OUT / 'cleaned_stock_prices.csv', index=False)

cleaning_summary = pd.DataFrame({
    'Measure': ['Rows before cleaning', 'Rows after cleaning', 'Duplicates removed',
                'Missing values before', 'Missing values after'],
    'Value': [len(stock_raw), len(stock), duplicates_removed,
              int(stock_raw.isna().sum().sum()), int(stock.isna().sum().sum())]
})
print(cleaning_summary)
print('Data types after cleaning:')
print(stock.dtypes.to_frame('dtype'))

# ## Task 2: Exploratory Data Analysis (EDA)
iris = pd.read_csv(DATA / 'iris.csv')
print('Shape:', iris.shape)
print('Missing values:', int(iris.isna().sum().sum()))
print('Duplicate rows:', int(iris.duplicated().sum()))
print(iris.head())

numeric_cols = iris.select_dtypes(include='number').columns.tolist()
summary_stats = iris[numeric_cols].agg(['mean','median','std','min','max']).T
summary_stats['mode'] = iris[numeric_cols].mode().iloc[0]
summary_stats = summary_stats[['mean','median','mode','std','min','max']].round(3)
summary_stats.to_csv(OUT / 'iris_summary_statistics.csv')
print(summary_stats)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), numeric_cols):
    ax.hist(iris[col], bins=15, edgecolor='black', alpha=0.8)
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
    ax.set_xlabel(col.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(OUT / 'iris_histograms.png', dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.boxplot(data=iris, x='species', y=col, ax=ax)
    ax.set_title(f'{col.replace("_", " ").title()} by Species')
    ax.set_xlabel('Species')
    ax.set_ylabel(col.replace('_', ' ').title())
plt.tight_layout()
plt.savefig(OUT / 'iris_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 6))
for species, group in iris.groupby('species'):
    plt.scatter(group['petal_length'], group['petal_width'], label=species, alpha=0.8)
plt.title('Petal Length versus Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig(OUT / 'iris_scatter_petal.png', dpi=200, bbox_inches='tight')
plt.close()

corr = iris[numeric_cols].corr().round(3)
corr.to_csv(OUT / 'iris_correlation_matrix.csv')
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix of Iris Measurements')
plt.tight_layout()
plt.savefig(OUT / 'iris_correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print(corr)

print('\nLevel 1 complete: data cleaning and EDA outputs saved to outputs/level1.')
