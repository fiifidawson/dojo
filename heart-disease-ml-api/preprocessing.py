# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# Loadin csv
data = pd.read_csv("./data/heart_disease.csv")

"""
Exploratory Data Analysis - Simple
"""

print("=" * 80)
# Print data sections
print(data.shape) # data shape
print(data.head(10)) # data head
print("=" * 80)

print("\n")
print("=" * 80)
# Check for missing values
print(data.isnull().sum())

if data.isnull().sum().sum() == 0:
    print("There is no case of missing data!")
else: 
    print(f"There is missing data {data.isnull().sum()}!")
print("=" * 80)

print("\n")
print("=" * 80)
# Patient info
print(data.info())
print("=" * 80)

print("\n")
print("=" * 80)
# Statistical measures about the data
print(data.describe())
print("=" * 80)

print("\n")
print("=" * 80)
"""Target - Prediction
Checking the distribution of target variable to see how many have heart disease and how many do not
"""
print(data['target'].value_counts())

# Splitting features and target
# feature, x
x = data.drop(columns='target')

# target, y
y = data['target']

## Splitting data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# print("Shapes")
# print(f'X train: {x_train.shape} | X test: {x_test.shape} | \ny train: {y_train.shape} | y test: {y_test.shape}')

print("\n")
print("=" * 80)
## Model Training
model = LogisticRegression()

model.fit(x_train, y_train)

## Model evaluation
# Training accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print(f"Accuracy on training data: {training_data_accuracy}")

# Test accuracy
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print(f"Accuracy on test data: {test_data_accuracy}")

print(confusion_matrix(y_test, x_test_prediction))
print(classification_report(y_test, x_test_prediction))

## Create models directory and save the models
os.makedirs('models', exist_ok=True)
with open('models/heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")