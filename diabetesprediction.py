import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
df = pd.read_csv(r"C:\Users\thenn\OneDrive\Desktop\diabetes.csv")

# Check for missing values
print(df.info())  # This helps verify data types and null values
print(df.isnull().sum())  # Counts missing values per column

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features and target variable
X = df.drop(columns=['Outcome'], axis=1)
y = df['Outcome']

# Standardize features before splitting to prevent data leakage
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X_standardized, y, test_size=0.2, random_state=2, stratify=y)

# Train SVM model
classifier_model = svm.SVC(kernel='linear')
classifier_model.fit(X_train, Y_train)

# Make predictions
predictions = classifier_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Example of making a single prediction
input_data = X_standardized[6].reshape(1, -1)
single_prediction = classifier_model.predict(input_data)
print(f"Single Prediction: {single_prediction[0]}")