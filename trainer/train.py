import os
import mlflow

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage0-practice")

# 항상 outputs 폴더 만들어두기
os.makedirs("outputs", exist_ok=True)

# 테스트용으로 더미 파일 생성 (없으면 log_artifact 에러 발생)
with open("outputs/loss_curve.png", "wb") as f:
    f.write(b"dummy content")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.95)
    mlflow.log_artifact("outputs/loss_curve.png")

    print("MLflow 실험 기록 완료!")