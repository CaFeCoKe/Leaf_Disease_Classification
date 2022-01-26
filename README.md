# Leaf_Disease_Classification
작물 잎 사진을 훈련 데이터와 테스트 데이터를 일정 비율로 나누어 훈련 데이터로 질병의 유무를 따져 분류하는 모델을 만들어 테스트 데이터로 모델에 대한 정확도를 측정한다.

https://user-images.githubusercontent.com/86700191/150724332-cde6c381-2a9b-4eec-883d-8844fa64277c.mp4

## 1. 사용 라이브러리
- PyTorch : 이미지 데이터 전처리 및 Tensor화, CNN Network 구성

## 2. 알고리즘 순서도
![Leaf_Classification](https://user-images.githubusercontent.com/86700191/151174087-52378e6c-ccb5-4af6-8f03-1cfc5757efa7.png)
## 3. 유의점
- 훈련, 테스트 데이터를 분할하여 저장할 폴더인 splitted 폴더가 없어야 한다.
- DataLoader의 파라미터 중 num_worker는 멀티프로세스의 수를 정하는 것인데 윈도우환경에서는 default값인 0 (메인프로세스 1개 사용)을 사용하지 않으면 BrokenPipeError가 발생할 수 있다.

## 4. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [Base Code](https://github.com/bjpublic/DeepLearningProject)
- [작물 잎 사진 원본데이터](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- [CNN에 대한 간단한 설명](https://yjjo.tistory.com/8)
- [Dropout에 대한 간단한 설명](https://heytech.tistory.com/127)