import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# 특징 추출 함수
def extract_features(file_path, sr=22050, n_mfcc=13, hop_length=512, segment_duration=2.0):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    max_len = int(sr * segment_duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    features = np.vstack([mfcc, zcr])
    return features.T  # (time, feature_dim)

#  전체 데이터 로드 (numpy 반환)
def load_data(base_dir='data'):
    class_map = {'non_noisy': 0, 'noisy': 1}
    X, y = [], []
    for class_name, label in class_map.items():
        class_dir = os.path.join(base_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                f_path = os.path.join(class_dir, file)
                features = extract_features(f_path)
                X.append(features.mean(axis=0))  # 평균값으로 고정길이 벡터
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# PyTorch Dataset 클래스 (바로 DataLoader에 쓸 경우)
class NoiseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
