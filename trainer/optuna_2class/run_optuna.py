import optuna
import mlflow
import os
from objective_fn import objective

# MLflow 서버 URI 및 experiment 설정
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage3-practical-project-v1")

#  Optuna study 생성 및 최적화 시작
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Best trial 출력
print("Best trial:")
print(study.best_trial)

# Best trial 결과 기록
os.makedirs("outputs", exist_ok=True)
best_trial_path = os.path.join("outputs", "best_trial.txt")
with open(best_trial_path, "w") as f:
    f.write(str(study.best_trial))

# Best 파라미터를 JSON으로도 저장 (후속 활용 쉽게)
import json
best_params_path = os.path.join("outputs", "best_params.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_trial.params, f, indent=4)

#  최종 결과를 MLflow에도 기록
with mlflow.start_run(run_name="optuna_summary"):
    mlflow.log_artifact(best_trial_path)
    mlflow.log_artifact(best_params_path)
