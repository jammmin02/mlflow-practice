import torch
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_loader import load_data
from model import NoiseClassifier

#  device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  best 파라미터 로드
with open("outputs/best_params.json", "r") as f:
    best_params = json.load(f)

hidden_dim = best_params['hidden_dim']
dropout_rate = best_params['dropout_rate']

#  데이터 로드
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  모델 생성 (튜닝된 파라미터로 반드시 생성)
model = NoiseClassifier(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
model.load_state_dict(torch.load("outputs/best_model.pt", map_location=device))
model.to(device)
model.eval()

#  추론
with torch.no_grad():
    inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

#  평가
acc = accuracy_score(y_test, preds)
print(f" PyTorch Inference Accuracy: {acc:.4f}")
