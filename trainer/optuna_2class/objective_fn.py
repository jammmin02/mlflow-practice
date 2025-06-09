import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model import create_model
from data_loader import load_data

#  reproducibility
torch.manual_seed(42)
np.random.seed(42)

#  데이터 로드 (전체에서 미리 분리해둠)
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    #  하이퍼파라미터 서치 공간
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    with mlflow.start_run(run_name=f"Trial {trial.number}", nested=True):
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        # 모델 생성
        model = create_model(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 텐서 변환
        train_inputs = torch.tensor(X_train, dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)

        # 간단한 DataLoader 형태로 batch 구성
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 간단한 10 epoch 훈련
        model.train()
        for epoch in range(10):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        # 검증
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_outputs = model(val_inputs)
            val_preds = torch.argmax(val_outputs, dim=1).numpy()

        acc = accuracy_score(y_val, val_preds)
        mlflow.log_metric("val_accuracy", acc)

        #  모델 저장 (trial 별로)
        torch.save(model.state_dict(), f"model_trial_{trial.number}.pt")
        mlflow.log_artifact(f"model_trial_{trial.number}.pt")

    return acc
