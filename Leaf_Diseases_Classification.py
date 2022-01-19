import os
import shutil
import math

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# 모델 학습을 위한 준비
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")    # GPU 존재시 GPU 실행(CUDA)

BATCH_SIZE = 256
EPOCH = 30

# 이미지 데이터를 64*64의 크기의 Tensor로 변환
transform_base = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
train_dataset = ImageFolder(root='./splitted/train', transform=transform_base)
val_dataset = ImageFolder(root='./splitted/val', transform=transform_base)

# Tensor화 된 이미지 데이터를 배치 사이즈로 분리(매 epoch마다 순서가 섞이며, 데이터 로딩에 서브프로세스 2개 사용)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# 네트워크 설계
class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()

        # 3-Convolution Layer
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layer
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 33)

        # 활성화 함수 ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolution+activation -> Pooling -> Dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)    # 25% 노드는 Dropout -> 과적합 방지 + 앙상블 비슷한 효과

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x, p=0.25, training=self.training)

        # fully connect+activation -> Dropout -> fully connect
        x = x.view(-1, 4096)    # fc를 위한 데이터 재배치
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)      # CNN 출력에 softmax 함수를 이용해 데이터가 각 클래스로 분류될 확률을 출력


cnn = CNN_Model().to(DEVICE)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)      # adam optimizer 사용, 학습률은 0.001
