import os
import shutil
import math
import time
import copy

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

# 훈련, 테스트 데이터를 저장할 기본폴더 생성
base_dir = './splitted'
os.mkdir(base_dir)

# splitted 폴더에 훈련, 테스트 데이터 폴더 생성
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 훈련, 테스트 데이터 폴더에 원본 데이터와 같은 하위폴더 생성
for cls in classes_list:
    os.mkdir(os.path.join(train_dir, cls))
    os.mkdir(os.path.join(test_dir, cls))

# 데이터 분할 및 클래스별 데이터 개수 확인
for cls in classes_list:
    path = os.path.join(original_dataset_dir, cls)
    fnames = os.listdir(path)

    # 파일 개수 비율 (훈련:테스트 = 7:3)
    train_size = math.floor(len(fnames) * 0.7)
    test_size = math.floor(len(fnames) * 0.3)

    # 파일 개수 표시
    train_fnames = fnames[:train_size]
    print("Train size(", cls, "): ", len(train_fnames))
    test_fnames = fnames[train_size:(test_size + train_size)]
    print("Test size(", cls, "): ", len(test_fnames))

    # 데이터 분할 저장
    for fname in train_fnames:
        src = os.path.join(path, fname)    # 복사할 파일의 경로 지정
        dst = os.path.join(os.path.join(train_dir, cls), fname)     # 붙여넣기할 경로 지정
        shutil.copyfile(src, dst)       # src 경로의 파일을 복사, dst 경로에 붙여넣기

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
test_dataset = ImageFolder(root='./splitted/test', transform=transform_base)

# Tensor화 된 이미지 데이터를 배치 사이즈로 분리(매 epoch마다 순서가 섞이며, 데이터 로딩에 서브프로세스 2개 사용)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


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


# 훈련 데이터로 학습하여 모델화 함수
def train(model, train_loader, optimizer):
    model.train()       # 훈련을 위해 Dropout 설정
    for batch_idx, (data, target) in enumerate(train_loader):   # train_loader 형태 = 배치 인덱스 (data, target) = 미니배치
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()       # optimizer 초기화
        output = model(data)
        loss = F.cross_entropy(output, target)      # Loss 함수로 교차 엔트로피 사용
        loss.backward()     # 역전파로 Gradient를 계산 후 파라미터에 할당
        optimizer.step()    # 파라미터 업데이트


# 모델 평가 함수
def evaluate(model, test_loader):
    model.eval()       # 평가를 위해 훈련과정에서 Dropout 한 노드설정을 해제
    test_loss = 0      # Loss 값 초기화
    correct = 0        # 올바르게 예측 된 데이터의 개수

    with torch.no_grad():       # Gradient 계산 비활성화 (모델 평가에는 파라미터 업데이트 X)
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()    # Loss 함수로 교차 엔트로피 사용

            pred = output.max(1, keepdim=True)[1]       # 33개의 클래스에 속할 확률값 중 가장 높은값의 인덱스를 예측값으로 지정
            correct += pred.eq(target.view_as(pred)).sum().item()       # target의 Tensor를 pred의 Tensor로 재정렬 후 비교하여 같으면 그 수를 합쳐 값 증가 (정확도 측정)

    test_loss /= len(test_loader.dataset)       # Loss 값을 Batch 값으로 나누어 미니 배치마다의 Loss 값의 평균을 구함
    test_accuracy = 100. * correct / len(test_loader.dataset)       # 정확도 값을 Batch 값으로 나누어 미니 배치마다의 정확도 평균을 구함
    return test_loss, test_accuracy


# 모델 학습 실행 함수
def execute_model(model, train_loader, test_loader, optimizer, num_epochs):
    best_acc = 0.0      # 가장 높은 정확도 저장
    best_model_wts = copy.deepcopy(model.state_dict())      # 가장 높은 정확도 모델 저장

    for epoch in range(1, num_epochs + 1):
        start = time.time()     # 한 epoch 시작 시각 저장
        train(model, train_loader, optimizer)   # 훈련데이터로 모델 학습
        train_loss, train_acc = evaluate(model, train_loader)   # 훈련 데이터 모델 Loss, 정확도 계산
        test_loss, test_acc = evaluate(model, test_loader)     # 테스트 데이터 모델 Loss, 정확도 계산

        if test_acc > best_acc:      # 현 epoch의 정확도가 더 높을 시 갱신
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time = time.time() - start      # epoch 걸린 시간 (시작시각 - 종료시각)

        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))

    model.load_state_dict(best_model_wts)   # 정확도가 가장 높은 모델을 불러오기
    return model


base = execute_model(cnn, train_loader, test_loader, optimizer, EPOCH)
torch.save(base, 'CNN_model.pt')        # 학습 완료된 모델 저장
