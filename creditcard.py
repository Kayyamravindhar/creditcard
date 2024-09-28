# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display basic information about the dataset
print("Dataset Overview: ")
print(data.head())

# Check for missing values
print("\nMissing Values: ")
print(data.isnull().sum())

# Drop any rows with missing values (if any)
data = data.dropna()

# Split data into features (X) and target (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable (fraud or not fraud)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nData After Applying SMOTE:")
print(f"Number of fraudulent transactions: {sum(y_train_res)}")
print(f"Number of genuine transactions: {len(y_train_res) - sum(y_train_res)}")

# Train a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred_logreg = logreg.predict(X_test)

# Evaluate Logistic Regression model
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Confusion Matrix for Logistic Regression
print("\nConfusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred_logreg))

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
print("\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# Visualize the confusion matrix for Random Forest
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Compute ROC curve and AUC for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()
