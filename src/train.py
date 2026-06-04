import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Features and target
X = df.drop(columns=["CustomerId", "is_high_risk"])
y = df["is_high_risk"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"\n{name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", roc_auc)


param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print(grid.best_params_)
with mlflow.start_run(run_name="Best_RandomForest"):

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    mlflow.log_params(grid.best_params_)

    print(grid.best_params_)


joblib.dump(
    best_model,
    "best_model.pkl"
)
