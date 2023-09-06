# Done By: Ahmed Nader Ali - 20186118
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Read CSV File
# Dataset Source: https://www.kaggle.com/datasets/thedevastator/predicting-heart-disease-risk-using-clinical-var
df = pd.read_csv("Heart_Disease_Prediction.csv")

# Fill the empty fields with NaN string.
df.replace('', np.nan, inplace=True)

# Drop all the rows that has NaN value.
df.dropna(inplace=True)

# Slicing the data into x and y
x = df.iloc[:, 1:13]
# Unification Process
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, -1])
df['Heart Disease'] = y

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Applying Z-Score Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create logistic regression model and fit it with our values
lr = LogisticRegression(fit_intercept=True)
lr.fit(X_train_scaled, y_train)
y_predicted = lr.predict(X_test_scaled)

# Print Results
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_predicted))
print("Logistic Regression Precision: ", precision_score(y_test, y_predicted))
print("Logistic Regression Recall: ", recall_score(y_test, y_predicted))
print("Logistic Regression F1-Score: ", f1_score(y_test, y_predicted))

# Create Naive Bayes Model and fit it with our values
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Predict values with Naive Bayes Model
y_pred_nb = nb.predict(X_test_scaled)

# Print Results
print("\nNaive Bayes Accuracy: ", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Precision: ", precision_score(y_test, y_pred_nb))
print("Naive Bayes Recall: ", recall_score(y_test, y_pred_nb))
print("Naive Bayes F1-Score: ", f1_score(y_test, y_pred_nb))