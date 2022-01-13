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

