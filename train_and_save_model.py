import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names.tolist()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline = scaling + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)

    # evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # save model and metadata
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": feature_names,
        "target_names": target_names,
        "metrics": {"accuracy": acc, "report": report}
    }, "model.pkl")

    print(f"Saved model.pkl  |  accuracy={acc:.3f}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
