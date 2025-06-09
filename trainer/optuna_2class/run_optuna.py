import optuna
import mlflow
import os
import json
import argparse
from objective_fn import objective

#  argparse 추가
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="stage3-practical-project-v1")
parser.add_argument("--n_trials", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="outputs")
args = parser.parse_args()

#  MLflow 서버 URI 및 experiment 설정
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment(args.experiment_name)

#  Optuna study 생성 및 최적화 시작
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.n_trials)

#  Best trial 출력
print("Best trial:")
print(study.best_trial)

#  Best trial 결과 기록
os.makedirs(args.output_dir, exist_ok=True)

best_trial_path = os.path.join(args.output_dir, "best_trial.txt")
with open(best_trial_path, "w") as f:
    f.write(str(study.best_trial))

best_params_path = os.path.join(args.output_dir, "best_params.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_trial.params, f, indent=4)

#  최종 결과를 MLflow에도 기록
with mlflow.start_run(run_name="optuna_summary"):
    mlflow.log_artifact(best_trial_path)
    mlflow.log_artifact(best_params_path)
