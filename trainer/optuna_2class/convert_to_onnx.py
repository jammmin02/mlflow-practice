import torch
import torch.onnx
import numpy as np
from model import create_model
from data_loader import load_data
from sklearn.model_selection import train_test_split

# 데이터 로드
X, y = load_data()
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# 넘파이 → 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# 모델 생성 (하이퍼파라미터는 적절히 수정 필요)
model = create_model()

# 간단히 한 번 학습 (에포크, 옵티마이저 등은 실제 학습과 다를 수 있음)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(10):  # 예시로 10 epoch만
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# ONNX 변환 준비
dummy_input = torch.randn(1, X_train.shape[1], dtype=torch.float32)

# ONNX로 변환
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    input_names=['float_input'], 
    output_names=['output'],
    dynamic_axes={'float_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

print("PyTorch 모델이 ONNX로 변환 완료되었습니다!")
