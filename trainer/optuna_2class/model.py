import torch
import torch.nn as nn

# 하이퍼파라미터 기반 모델 생성 함수
def create_model(input_dim=14, hidden_dim=64, output_dim=2, dropout_rate=0.3):
    model = NoiseClassifier(input_dim, hidden_dim, output_dim, dropout_rate)
    return model

#  모델 클래스 정의
class NoiseClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=2, dropout_rate=0.3):
        super(NoiseClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)
