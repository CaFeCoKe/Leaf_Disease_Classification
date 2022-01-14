import os
import shutil
import math

# 원본 데이터셋 연결
original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir)

# 훈련, 평가, 테스트 데이터를 저장할 기본폴더 생성
base_dir = './splitted'
os.mkdir(base_dir)

# splitted 폴더에 훈련, 평가, 테스트 데이터 폴더 생성
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 훈련, 평가, 테스트 데이터 폴더에 원본 데이터와 같은 하위폴더 생성
for cls in classes_list:
    os.mkdir(os.path.join(train_dir, cls))
    os.mkdir(os.path.join(validation_dir, cls))
    os.mkdir(os.path.join(test_dir, cls))

# 데이터 분할 및 클래스별 데이터 개수 확인
for cls in classes_list:
    path = os.path.join(original_dataset_dir, cls)
    fnames = os.listdir(path)

    # 파일 개수 비율 (훈련:평가:테스트 = 6:2:2)
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) * 0.2)
    test_size = math.floor(len(fnames) * 0.2)

    # 파일 개수 표시
    train_fnames = fnames[:train_size]
    print("Train size(", cls, "): ", len(train_fnames))
    validation_fnames = fnames[train_size:(validation_size + train_size)]
    print("Validation size(", cls, "): ", len(validation_fnames))
    test_fnames = fnames[(train_size + validation_size):(validation_size + train_size + test_size)]
    print("Test size(", cls, "): ", len(test_fnames))

    # 데이터 분할 저장
    for fname in train_fnames:
        src = os.path.join(path, fname)    # 복사할 파일의 경로 지정
        dst = os.path.join(os.path.join(train_dir, cls), fname)     # 붙여넣기할 경로 지정
        shutil.copyfile(src, dst)       # src 경로의 파일을 복사, dst 경로에 붙여넣기

    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(validation_dir, cls), fname)
        shutil.copyfile(src, dst)

    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, cls), fname)
        shutil.copyfile(src, dst)
