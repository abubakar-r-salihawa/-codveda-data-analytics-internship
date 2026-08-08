# Codveda Data Analytics Internship

**Intern:** Abubakar Rabiu Salihawa  
**Intern ID:** CV/AI/82271  
**Role:** Data Analysis Intern

This repository contains the completed Codveda data analytics internship tasks. Two tasks were completed from each level, following the task brief.

## Completed tasks

### Level 1 — Basic
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)

### Level 2 — Intermediate
1. Regression Analysis
2. Time Series Analysis

### Level 3 — Advanced
1. Predictive Modelling (Classification)
2. Natural Language Processing — Sentiment Analysis

## Repository structure

- `Level_1_Basic/` — data cleaning and EDA notebook
- `Level_2_Intermediate/` — regression and time-series notebook
- `Level_3_Advanced/` — classification and sentiment-analysis notebook
- `data/` — datasets required to reproduce the notebooks
- `outputs/` — generated tables and figures after running the notebooks
- `requirements.txt` — Python dependencies

## How to run

1. Clone the repository.
2. Install the packages in `requirements.txt`.
3. Open each notebook from its level folder.
4. Run the cells from top to bottom.

The notebooks use relative paths and fixed random states where appropriate so that the main modelling results can be reproduced.

## Note on the stock-price dataset

The original stock-price file supplied for the internship is about 24 MB. To keep this repository lightweight, `data/stock_prices.csv` contains the AAPL observations used for the time-series task together with the rows containing missing price values used to demonstrate the Level 1 cleaning workflow. The notebook logic is unchanged and can also be run against the full source file if it is placed at the same path.

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, NLTK and TextBlob.
