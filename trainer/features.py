import librosa
import numpy as np

def extract_features(file_path, sr=22050, n_mfcc=13, hop_length=512, segment_duration=2.0):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # 오디오 길이 조정
    max_len = int(sr * segment_duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    # 특징 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    # (n_mfcc + zcr) 통합
    features = np.vstack([mfcc, zcr])
    return features.T