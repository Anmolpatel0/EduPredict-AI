"""Train and evaluate the Student Performance Decision Tree from the terminal."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

rng = np.random.default_rng(42)
n = 300
study_hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)
previous_marks = rng.uniform(20, 100, n)
score = study_hours * 4 + attendance * 0.30 + previous_marks * 0.25 + rng.normal(0, 5, n)
result = (score >= 55).astype(int)

df = pd.DataFrame({
    "Study_Hours": np.round(study_hours, 2),
    "Attendance": np.round(attendance, 2),
    "Previous_Marks": np.round(previous_marks, 2),
    "Result": result
})
df.to_csv("dataset.csv", index=False)

X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Dataset saved to dataset.csv")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
print(classification_report(y_test, predictions, target_names=["Fail", "Pass"]))
