#!/usr/bin/env python
# coding: utf-8

# # Codveda Data Analytics Internship - Level 3 (Advanced)
#
# **Intern:** Abubakar Rabiu Salihawa
# **Internship:** Data Analysis Intern, Codveda Technologies
# **Intern ID:** CV/AI/82271
#
# Selected tasks: Task 1 - Predictive Modeling (Classification) and Task 3 - NLP Sentiment Analysis.

from pathlib import Path
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.stem import PorterStemmer

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
OUT = ROOT / 'outputs' / 'level3'
OUT.mkdir(parents=True, exist_ok=True)

# Task 1: Predictive Modeling (Classification)
train = pd.read_csv(DATA / 'churn_train_80.csv')
test = pd.read_csv(DATA / 'churn_test_20.csv')
print('Training shape:', train.shape)
print('Testing shape:', test.shape)
print('Training churn rate:', f"{train['Churn'].mean():.2%}")
print('Testing churn rate:', f"{test['Churn'].mean():.2%}")

X_train = train.drop(columns='Churn')
y_train = train['Churn'].astype(int)
X_test = test.drop(columns='Churn')
y_test = test['Churn'].astype(int)

categorical = X_train.select_dtypes(include='object').columns.tolist()
numerical = X_train.select_dtypes(exclude='object').columns.tolist()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=1)
}

results = []
for name, estimator in models.items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1-score': f1_score(y_test, pred, zero_division=0)
    })

results_df = pd.DataFrame(results).sort_values('F1-score', ascending=False)
results_df.to_csv(OUT / 'classification_model_comparison.csv', index=False)
print(results_df.round(4))

rf_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1))
])
param_grid = {
    'model__n_estimators': [150, 250],
    'model__max_depth': [8, 14],
    'model__min_samples_leaf': [1, 2]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(rf_pipe, param_grid, scoring='f1', cv=cv, n_jobs=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)
tuned_metrics = pd.DataFrame({
    'Metric': ['Accuracy','Precision','Recall','F1-score','Best CV F1-score'],
    'Value': [accuracy_score(y_test,best_pred), precision_score(y_test,best_pred),
              recall_score(y_test,best_pred), f1_score(y_test,best_pred), grid.best_score_]
})
tuned_metrics.to_csv(OUT / 'tuned_random_forest_metrics.csv', index=False)
print('Best parameters:', grid.best_params_)
print(tuned_metrics.round(4))
print(classification_report(y_test, best_pred, target_names=['Stayed','Churned']))

ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=['Stayed','Churned'], cmap='Blues')
plt.title('Tuned Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig(OUT / 'churn_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()

feature_names = best_model.named_steps['preprocess'].get_feature_names_out()
importance = best_model.named_steps['model'].feature_importances_
importance_df = (pd.DataFrame({'Feature': feature_names, 'Importance': importance})
                 .sort_values('Importance', ascending=False).head(15))
importance_df.to_csv(OUT / 'churn_top_feature_importance.csv', index=False)

plt.figure(figsize=(10, 7))
plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
plt.title('Top 15 Features Influencing Churn Prediction')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig(OUT / 'churn_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()

# Task 3: NLP Sentiment Analysis
sent = pd.read_csv(DATA / 'sentiment.csv')
sent = sent.drop(columns=[c for c in sent.columns if c.lower().startswith('unnamed')], errors='ignore')
for col in sent.select_dtypes(include='object').columns:
    sent[col] = sent[col].astype(str).str.strip()
sent['Timestamp'] = pd.to_datetime(sent['Timestamp'], errors='coerce')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

sent['clean_text'] = sent['Text'].map(clean_text)
stopwords = set('a an the and or but if is are was were be been being to of in on for with at by from this that these those it its i you he she they we my your our their as so just very have has had do does did not no'.split())
stemmer = PorterStemmer()
sent['tokens'] = sent['clean_text'].map(lambda x: [w for w in x.split() if len(w) > 2 and w not in stopwords])
sent['stemmed_tokens'] = sent['tokens'].map(lambda words: [stemmer.stem(w) for w in words])
sent['polarity'] = sent['clean_text'].map(lambda x: TextBlob(x).sentiment.polarity)

def classify(p):
    if p > 0.05:
        return 'Positive'
    if p < -0.05:
        return 'Negative'
    return 'Neutral'

sent['predicted_sentiment'] = sent['polarity'].map(classify)
sent.to_csv(OUT / 'sentiment_analysis_results.csv', index=False)

sentiment_counts = sent['predicted_sentiment'].value_counts().reindex(['Positive','Neutral','Negative'], fill_value=0)
sentiment_counts.to_csv(OUT / 'sentiment_distribution.csv', header=['count'])
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title('TextBlob Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.tight_layout()
plt.savefig(OUT / 'sentiment_distribution.png', dpi=200, bbox_inches='tight')
plt.close()

all_words = [w for words in sent['tokens'] for w in words]
word_counts = pd.DataFrame(Counter(all_words).most_common(25), columns=['word','frequency'])
word_counts.to_csv(OUT / 'top_word_frequencies.csv', index=False)
plt.figure(figsize=(10, 7))
plt.barh(word_counts['word'][::-1], word_counts['frequency'][::-1])
plt.title('Top 25 Words in Social-Media Posts')
plt.xlabel('Frequency')
plt.tight_layout()
plt.savefig(OUT / 'top_word_frequencies.png', dpi=200, bbox_inches='tight')
plt.close()

cloud = WordCloud(width=800, height=400, background_color='white', collocations=False, max_words=100, random_state=42).generate(' '.join(all_words))
plt.figure(figsize=(10, 5))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Cleaned Social-Media Text')
plt.tight_layout()
plt.savefig(OUT / 'sentiment_wordcloud.png', dpi=200, bbox_inches='tight')
plt.close()

print('\nLevel 3 complete: classification and NLP outputs saved to outputs/level3.')
