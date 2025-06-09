import mlflow
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from features import extract_features

# 서버 URI (docker-compose 네트워크 명)
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("stage1-noise-experiment")

# 데이터 경로
base_dir = 'data'
class_map = {'non_noisy': 0, 'noisy': 1}

# 데이터 로드
X, y = [], []
for class_name, label in class_map.items():
    class_dir = os.path.join(base_dir, class_name)
    for file in os.listdir(class_dir):
        if file.endswith('.wav') or file.endswith('.mp3'):
            f_path = os.path.join(class_dir, file)
            features = extract_features(f_path)
            X.append(features.mean(axis=0))  # 평균통계 사용 (간단버전)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Train/Val 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터
C = 1.0

with mlflow.start_run():
    # 기록 시작
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("C", C)

    # 모델 학습
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)

    # 검증
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    # 기록
    mlflow.log_metric("val_accuracy", acc)

    # 아티팩트 저장 예시
    np.save("model_weights.npy", model.coef_)
    mlflow.log_artifact("model_weights.npy")

    print(f"Validation Accuracy: {acc}")