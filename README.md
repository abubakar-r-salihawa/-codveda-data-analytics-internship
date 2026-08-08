# Codveda Data Analytics Internship

**Intern:** Abubakar Rabiu Salihawa  
**Intern ID:** CV/AI/82271  
**Role:** Data Analysis Intern

This repository contains my completed Codveda Data Analytics internship tasks. Two tasks were completed from each level in line with the internship task brief.

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

- `Level_1_Basic/Level_1_Data_Cleaning_and_EDA.ipynb` — data cleaning and exploratory analysis
- `Level_2_Intermediate/Level_2_Regression_and_Time_Series.ipynb` — regression and time-series analysis
- `Level_3_Advanced/Level_3_Classification_and_NLP.ipynb` — classification and sentiment analysis
- `data/README.md` — dataset names and expected file locations
- `requirements.txt` — Python dependencies

The three levels are kept separately so each stage of the internship can be reviewed independently. The notebooks combine short explanations, code, results and visualisations in one place.

## How to run

1. Clone or download the repository.
2. Install the packages listed in `requirements.txt`.
3. Place the supplied internship datasets in the `data/` folder using the filenames listed in `data/README.md`.
4. Open the relevant `.ipynb` file in Jupyter Notebook, JupyterLab, VS Code or Google Colab.
5. Run the notebook cells from top to bottom.
6. Generated tables and figures are saved under `outputs/level1`, `outputs/level2`, and `outputs/level3`.

Relative paths are used so the notebooks can be reproduced without changing the code when the repository structure is kept unchanged. Fixed random states are used where appropriate for repeatable modelling results.

## Project note

The original datasets remain with the internship submission package because they were supplied specifically for the exercise and include a large stock-price file. The notebooks preserve the required workflow and can be executed against the supplied source files using the documented filenames.

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, NLTK, TextBlob and WordCloud.
