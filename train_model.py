"""Train the XGBoost pipeline and save it to model.joblib.

Usage:  python train_model.py path/to/Predictive_Maintenance_1.1.csv
"""
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = ["Air temperature", "Process temperature",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = ["Type"]

csv_path = sys.argv[1] if len(sys.argv) > 1 else "Predictive_Maintenance_1.1.csv"
df = pd.read_csv(csv_path).drop(columns=["date", "device", "Product ID"])
df = df.drop_duplicates()

X = df[NUMERIC + CATEGORICAL]
y = df["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(categories=[["H", "L", "M"]],
                                               handle_unknown="ignore"))]), CATEGORICAL),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(random_state=42, eval_metric="logloss")),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
