# Codveda Data Analytics Internship

**Intern:** Abubakar Rabiu Salihawa  
**Intern ID:** CV/AI/82271  
**Role:** Data Analysis Intern

This repository contains the completed Codveda Data Analytics internship project code. Two tasks were completed from each level, following the internship task brief.

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

- `Level_1_Basic/` — Python analysis for data cleaning and EDA
- `Level_2_Intermediate/` — Python analysis for regression and time series
- `Level_3_Advanced/` — Python analysis for classification and sentiment analysis
- `data/README.md` — dataset names and expected file locations
- `requirements.txt` — Python dependencies

The three levels are kept separately so each stage of the internship can be reviewed independently.

## How to run

1. Clone the repository.
2. Install the packages in `requirements.txt`.
3. Place the supplied internship datasets in the `data/` folder using the filenames listed in `data/README.md`.
4. Run each Python file from its corresponding level folder.
5. Generated tables and figures will be saved automatically under `outputs/level1`, `outputs/level2`, and `outputs/level3`.

The scripts use relative paths and fixed random states where appropriate so the main modelling results can be reproduced.

## Project notes

The repository focuses on the analysis code and methodology. The original datasets remain with the final internship submission package because they were supplied specifically for the internship exercise and include a large stock-price file. The scripts preserve the required workflow and can be executed against those source files without changing their paths.

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, NLTK, TextBlob and WordCloud.
