import mlflow
import optuna
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from features import extract_features

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage2-optuna-experiment")

base_dir = 'data'
class_map = {'non_noisy': 0, 'noisy': 1}

X, y = [], []
for class_name, label in class_map.items():
    class_dir = os.path.join(base_dir, class_name)
    for file in os.listdir(class_dir):
        if file.endswith('.wav') or file.endswith('.mp3'):
            f_path = os.path.join(class_dir, file)
            features = extract_features(f_path)
            X.append(features.mean(axis=0))
            y.append(label)

X = np.array(X)
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)

    with mlflow.start_run(run_name=f"Trial {trial.number}", nested=True):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)

        model = LogisticRegression(C=C)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        mlflow.log_metric("val_accuracy", acc)

        np.save("model_weights.npy", model.coef_)
        mlflow.log_artifact("model_weights.npy")

    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best trial:")
print(study.best_trial)

