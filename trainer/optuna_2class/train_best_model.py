import mlflow
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model import create_model
from data_loader import load_data

def main():
    #  MLflow 설정
    mlflow.set_tracking_uri("http://mlflow_server:5000")
    mlflow.set_experiment("stage3-practical-project-v1")

    #  데이터 로드
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Optuna best 파라미터 불러오기
    with open("outputs/best_params.json", "r") as f:
        best_params = json.load(f)

    hidden_dim = best_params['hidden_dim']
    dropout_rate = best_params['dropout_rate']
    lr = best_params['lr']
    batch_size = best_params['batch_size']

    #  출력 폴더 준비
    os.makedirs("outputs", exist_ok=True)

    with mlflow.start_run(run_name="best_model_training"):
        #  파라미터 기록
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        #  모델 생성
        model = create_model(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        #  Tensor 변환
        train_inputs = torch.tensor(X_train, dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #  학습 루프 (에포크 수는 실험 가능)
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        #  검증
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_outputs = model(val_inputs)
            val_preds = torch.argmax(val_outputs, dim=1).numpy()

        acc = accuracy_score(y_val, val_preds)
        mlflow.log_metric("val_accuracy", acc)

        #  모델 저장
        torch.save(model.state_dict(), "outputs/best_model.pt")
        mlflow.log_artifact("outputs/best_model.pt")

    print(" 최종 학습 및 MLflow 기록 완료!")

if __name__ == "__main__":
    main()
