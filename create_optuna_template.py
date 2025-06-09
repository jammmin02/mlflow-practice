import os

# 생성할 디렉토리와 파일 목록
target_dir = "trainer/optuna_2class"
files = [
    "analyze_trials.py",
    "convert_to_onnx.py",
    "data_loader.py",
    "evaluate_best_model.py",
    "model.py",
    "objective_fn.py",
    "run_optuna.py",
    "train_best_model.py"
]

# 디렉토리 생성
os.makedirs(target_dir, exist_ok=True)

# 각 파일 생성
for file_name in files:
    file_path = os.path.join(target_dir, file_name)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            # 기본 템플릿 주입 가능 (지금은 공백으로 생성)
            f.write(f"# {file_name}\n\n")
        print(f"생성 완료: {file_path}")
    else:
        print(f"⚠ 이미 존재: {file_path}")

print("\n모든 파일 생성 완료!")
