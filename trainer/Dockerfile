FROM python:3.9-slim

# 시스템 패키지 설치 (git, ffmpeg 등: librosa, mlflow, onnx 호환 위해)
RUN apt-get update && apt-get install -y git ffmpeg

# 파이썬 패키지 한 방에 설치
RUN pip install --no-cache-dir \
    mlflow==2.2.2 \
    optuna==3.5.0 \
    librosa==0.10.1 \
    scikit-learn==1.3.0 \
    matplotlib \
    tqdm \
    numpy \
    pandas \
    onnx \
    soundfile \
    torch

# 작업 디렉토리 설정
WORKDIR /app

# 소스코드 복사
COPY . .

# 기본 shell 실행 (원하면 나중에 entrypoint 교체 가능)
CMD ["bash"]
