# mixup: BEYOND EMPIRICAL RISK MINIMIZATION

# Index

- [Abstraction](#Abstraction)
- [Introduction](#Introduction)
- [Contribution](#Contribution)
- [From Empirical Risk Minimization To mixup](#From-Empirical-Risk-Minimization-To-mixup)
- [Experiments](#Experiments)   
    - [1. ImageNet classification](#1-imagenet-classification)
    - [2. CIFAR-10 and CIFAR-100](#2-cifar-10-and-cifar-100)
    - [3. Speech Data](#3-speech-data)
    - [4. Memorization of Corrupted Labels](#4-memorization-of-corrupted-labels)
    - [5. Robustness to Adversarial Examples](#5-robustness-to-adversarial-examples)
    - [6. Tabular Data](#6-tabular-data)
    - [7. Stabilization of Generative Adversarial Networks (GANs)](#7-stabilization-of-generative-adversarial-networks)

# Abstraction

**mixup**은 data와 label을 **convex combination**해서 새로운 데이터를 생성

📎 convex combination

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7a1e16d3-8372-45b5-804f-cdaaa3ba0954/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T103854Z&X-Amz-Expires=86400&X-Amz-Signature=385766567c57820522284231cb403dbdd526ed645b796657926a6a550c34ca7f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="20%" height="20%">

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4d741191-0ad2-46ea-863a-e0bec5a3504c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104015Z&X-Amz-Expires=86400&X-Amz-Signature=45029044c63d9430eab694a3993898566561c56d0db4705d35e3fbe0eed35a90&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

점들을 선형 결합(linear combination)할 때

→ 계수가 양수 + 계수의 합을 1로 제한

→ (그림) 주어진 지점을 서로 연결한 도형 안에 있는 P와 같은 지점들

# Introduction

### 신경망의 2가지 특징

1️⃣ ERM(Empirical Risk Minimization)

→ 훈련 데이터에 대한 평균 에러 최소화 ⇒ 신경망 최적화 

2️⃣ **SOTA**는 훈련 데이터의 크기에 선형적으로 비례하게 신경망의 크기가 커짐

📎 SOTA

SOTA는 State-of-the-art의 약자로 사전학습된 신경망들 중 현재 최고 수준의 신경망이라는 뜻이다

### ERM 기반 학습의 단점

- classical result (Vapnik & Chervonenkis, 1971)에 따르면, 모델의 크기가 훈련 데이터 수에 따라 비례해서 증가하지 않아야 ERM의 수렴이 보장 → 신경망의 2가지 특징과 모순
- (강한 규제(regularization) 방법을 써도) 훈련 데이터를 기억(memorize) ⇒ 과적합 현상 발생

📎 memorization

모델이 학습을 진행할 때, 정답만을 기억하고 내리는 행동
즉, 데이터 분포를 학습하는 것이 아니라 해당 데이터가 어떤 라벨에 해당하는지 기억하게 되는 것
⇒ test distribution에 대한 generalization 불가능

### VRM(Vicinal Risk Minimization)의 등장

- ERM의 차이점

훈련 데이터 + **훈련 데이터 근방의(vicinal) 분포까지도 학습** ⇒ 추가적인 data에 대해 결론 도출 가능

📎 훈련 데이터 근방의 분포로 어떻게 학습하는가

가상의 데이터를 근방의 분포로부터 샘플링 ⇒ 이를 학습에 이용

인접한 examples는 같은 클래스에 속한다고 가정

- 근방의 분포를 얻는 방법

→ 데이터 증강(Data Augmentation)을 사용

 ⇒ VRM 기반의 학습 ⊂ 데이터 증강을 통한 학습 

- 결론

데이터 증강(Data augmentation) → 신경망의 일반화(generalization)에 좋다 (= 과적합 ↓)

📎 Data Augmentation

적은 양의 데이터를 바탕으로 다양한 알고리즘을 통해 데이터의 양을 늘리는 기술

# Contribution

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bbfcce92-c3aa-4b5c-b439-93e9aa26a7e9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104059Z&X-Amz-Expires=86400&X-Amz-Signature=899d7aca7a0fb9bec3574b6a3b073b4eea1c9ca613f663108e5d0005487f77c5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

(단, (![image](https://latex.codecogs.com/gif.latex?x_i%2C%20y_i))과 (![image](https://latex.codecogs.com/gif.latex?x_j%2C%20y_j))는 training data에서 랜덤으로 추출한 example)

### mixup의 training 분포 확장 원리

📎 선형 보간법(linear interpolation)

통계적 혹은 실험적으로 구해진 데이터로부터, 주어진 데이터를 만족하는 근사 함수(f(x))를 구하고, 이 식을 이용하여 주어진 변수에 대한 함수값을 구하는 과정 

ex. (0, 0), (1, 10), (2, 20)이 주어졌을 때, 이들에 대한 근사 함수를 f(x) = 10x로 구하고, 1.5에 대한 함수 값으로 15를 구하는 것입니다.

### mixup의 장점

1️⃣ CIFAR-10, CIFAR100(실험2) 및 ImageNet-2012(실험1) 이미지 분류 데이터 셋에서 성능 좋음

2️⃣ 손상된 레이블로부터 학습하거나(실험4) adversarial example에서도(실험5) 강력함

3️⃣ 음성 데이터(실험3)와 표 형식(실험6) 데이터에 대한 일반화(generalization)를 개선하고, GAN(실험7)의 훈련을 안정화하는 데 사용 가능

### CIFAR-10 실험에 필요한 소스 코드

https://github.com/facebookresearch/mixup-cifar10

# From Empirical Risk Minimization To mixup

### ERM 기반의 학습

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9762a269-0bff-47b1-9e55-752097609f70/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104134Z&X-Amz-Expires=86400&X-Amz-Signature=1663c2ec24c68ce8c52dd39bacf8f828a4f3c8bd492e5f154dc4fcb66ca9b1d9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### f∈F

random feature vector X와 random target vector Y 사이의 관계를 설명하는 함수(≒ 딥러닝 모델)

### ![image](https://latex.codecogs.com/gif.latex?l)

f(x)와 y의 차이를 모델링하는 손실 함수

⇒ **전체 훈련 데이터에 대한 손실 함수의 평균**을 최소화시켜서 f(= (x,y) 사이의 관계를 모델링하는 함수)를 찾는다

📎 전체 훈련 데이터 분포에서 손실 함수에 대한 평균을 정의

→ 적분을 활용 ⇒ 손실 함수의 기댓값 정의 

### R(f)

기대 위험

→ 기대 위험을 계산하기 위해서는 P(x,y)를 알고 있어야 하지만, **데이터의 분포 P**를 모름

📎 훈련 데이터 D로부터 데이터의 분포 P를 근사

훈련 데이터를 ![image](https://latex.codecogs.com/gif.latex?D%3D%7B%28x_i%2C%20y_i%29%7D_%7Bi%3D1%7D%5En), 라고 했을 때, ![image](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29%5Csim%20P)를 따른다고 하자

D를 기반으로 경험적 분포(Empirical Distribution)을 정의하면, 

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1976a8d8-0ee3-420b-be0e-e00750b32a4e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104217Z&X-Amz-Expires=86400&X-Amz-Signature=bf69a342f629a9ed95fb89f374c16ac7a63f977ea3be6d28d29424eb3952d6c8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

📎 경험적 분포

반복된 시행을 통해 확률 변수가 일정 값을 넘지 않을 확률을 유추하는 함수

### ![image](https://latex.codecogs.com/gif.latex?%5Cdelta%20%28x_%3Dx_i%2C%20y%3Dy_i%29)

다이락-델타 함수(Dirac-Delta function)

→ ![image](https://latex.codecogs.com/gif.latex?%28x_%3Dx_i%2C%20y%3Dy_i%29)에서만 1이고 나머지 점에서는 0인 함수

⇒ 이를 통해서 기대 위험(Expected Risk)를 도출

📎 기대 위험(Expected Risk)

우리가 알지 못하는 실제 데이터 분포를 일반화할 때 발생하는 error의 기대치

### **기대 위험(Expected Risk)**

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6bd3b2e3-50d9-442d-b9bd-743615825898/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104248Z&X-Amz-Expires=86400&X-Amz-Signature=2f87980c1314b5c7271da3a1e699815eee6a8f5200f7ed7ec2958074bb3d4609&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

→ ![image](https://latex.codecogs.com/gif.latex?R_%5Cdelta%20%28f%29)를 최소화 ⇒ ERM 기반의 학습

→ 딥러닝과 같이 파라미터 수가 많은 모델을 학습할 경우, empirical risk를 최소화하는 과정에서 훈련 데이터를 기억(memorize)할 수 있다 (단점)

→ VRM 기반의 학습으로 확장시킬 필요가 있다

📎 수식

![image](https://latex.codecogs.com/gif.latex?%5Cdelta%20%28x_%3Dx_i%2C%20y%3Dy_i%29)에 의해, i번째 항만 남고 나머지는 0으로 사라짐

### VRM 기반의 학습

데이터의 분포 P에 VRM을 적용하면, P를 아래와 같이 나타낼 수 있다 (단, v는 근방의 분포)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4eb03ee7-3d9c-4408-bf84-d4caca5cf77a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104308Z&X-Amz-Expires=86400&X-Amz-Signature=2b929b829b7878b1a7ddfc9194c5931f0f2a83713448686c2a9ffd6f14d355c8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### ![image](https://latex.codecogs.com/gif.latex?V%28%5Ctilde%7Bx%7D%2C%20%5Ctilde%7By%7D%7Cx_i%2C%20y_i%29)

근방의 분포(vicinal distribution)

훈련 데이터의 feature-target 쌍 ![image](https://latex.codecogs.com/gif.latex?%28x%3Dx_i%2C%20y%3Dy_i%29) 근방에서 가상의 feature-target 쌍을 찾을 확률의 분포

![image](https://latex.codecogs.com/gif.latex?%28%5Ctilde%7Bx%7D%2C%20%5Ctilde%7By%7D%29)→ (Data Augmentation으로부터 만들어진) 가상의 feature-target vector

![image](https://latex.codecogs.com/gif.latex?%28x%3Dx_i%2C%20y%3Dy_i%29) → 훈련 데이터의 feature-target 쌍

### 새로운 기대 위험

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/72fa9b71-0fce-456f-b1bb-6638b1778827/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104330Z&X-Amz-Expires=86400&X-Amz-Signature=144361a9340584a5c0299c6f2d6af1b62eefc8987d92c4805c5bbba64607a9b1&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

→ ![image](https://latex.codecogs.com/gif.latex?R_%5Cnu%20%28f%29) 를 최소화 ⇒ VRM 기반의 학습

### mixup

vicinal distribution을 일반화한 식을 제시 ⇒ mixup 데이터 증강 기법 ⊂ VRM 학습

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/32b367e8-e7b2-4f17-8790-4d06e08d2b0f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104419Z&X-Amz-Expires=86400&X-Amz-Signature=8aaec21282af52ed15f49ef3571062bb143040afe7e6263baf5dd88a1aff5796&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

→ 두 데이터![image](https://latex.codecogs.com/gif.latex?%28%28x_i%2C%20y_i%29%2C%20%28x_j%2C%20y_j%29%29)를  λ를 이용해서 적절하게 섞은 후 모델의 학습에 활용

→ λ(lambda) 값은 **Beta(a,a) 분포**를 따른다 

mixup 모델은 모든 훈련 데이터셋을 통해 vicinity를 구함

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/46bc6fb4-a522-44e0-a795-047061417d4a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104440Z&X-Amz-Expires=86400&X-Amz-Signature=bf3898248049b64e83f3fbd2a37d0fe0b6a8d67c984471c9360c16ee35cc90cf&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="30%" height="30%">

(단, ![image](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29)과 ![image](https://latex.codecogs.com/gif.latex?%28x_j%2C%20y_j%29)는 training data에서 랜덤으로 추출한 example)

→ vicinal distribution에서 virtual feature-target vector를 추출

📎 베타 분포

확률에 대한 확률 분포

성공 횟수(α−1)와 실패 횟수(β−1)가 고정된 것이고, 확률이 확률변수

cf. 이항 분포

확률 p가 고정된 것이고, 성공 횟수 및 실패 횟수가 확률변수

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/054e594f-0853-4cab-aeb9-33062bea3031/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104515Z&X-Amz-Expires=86400&X-Amz-Signature=dda02d3c0c98da2b13d6e885874bf718a1cb7029d5b621024bd202a04f1a1ce2&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### Figure 1 (a)


<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/89eb0a30-97cb-4a5e-8653-8b2351873b43/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104530Z&X-Amz-Expires=86400&X-Amz-Signature=15a1ed9d5a3bb14a0547503b996c306f9d4df30ed22a74a022260157b41fbde0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

- Pytorch로 구현한 코드

### Figure 1 (b)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/35677c91-0bf0-444a-b3bd-d633aa0ae244/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T105139Z&X-Amz-Expires=86400&X-Amz-Signature=6cdcfe7452ca54f7520d740b36b6edbb3149f154c5b8f5073ee4e1adacc77e8b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

- ERM(좌측) VS. mixup(우측)

→ 파란색 부분은 해당 데이터 x가 주어졌을 때, 클래스가 1일 확률

→ ERM을 보면 파란색 부분이 클래스 1가 가까운 것과 먼 것 사이의 차이를 나타내지 못함
↔ mixup은 가까운 부분은 더 짙은 파란색으로 나타내어, uncertainty를 부드럽게(smoother) 표시

→ ERM은 두 클래스 간의 decision boundary가 뚜렷 ↔ mixup은 decision boundary가 부드러움 

⇒ mixup이 ERM에 비해 과적합이 덜 발생한다 + mixup이 **regularization** 역할 

📎 regularization

훈련 데이터에만 특화되지 않고 일반화(generalization)가 가능하도록 규제(penalty)를 가하는 기법

### what is mixup doing?

모델이 training data 간에 선형적인 양상 (linear behavior)을 가지게끔 해준다 

⇒ 훈련 데이터가 아닌 데이터에서 예상치 못한 결과(undesirable oscillations)를 내는 것을 줄여준다

(생각) 수식에서 x와 y를 λ와 함께 선형의 식으로 표현한 부분을 말하는 것 같다 → 선형의 식으로 표현해서 훈련 데이터 외에 새로운 데이터를 만들어 낼 수 있음 ⇒ 과적합 피하게 된다   

### Figure 2 (a)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e7915823-841d-4231-b90d-17842544abee/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104547Z&X-Amz-Expires=86400&X-Amz-Signature=83ab9c5fa350c1b2ad7adb7a739dfb239b94c3e04999d858d6d661750094d6ac&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

→ model prediction

→ 성능 : ERM < mixup

### Figure 2 (b)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a1e0c0de-dd38-4871-8adf-3139a2328214/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104601Z&X-Amz-Expires=86400&X-Amz-Signature=738d217591f6168db71d55e2bec65880464ba5059abc3a4f5a955467aa2003a9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

→ 훈련 데이터 간의 **gradient norm → 작게 나타날수록 더 안정적인 학습** 

→ 안정적인 학습 : ERM < mixup 

📎 gradient norm

많이 틀릴 수록 기울기가 가팔라지므로, 크게 나타날수록 학습이 불안정

# Experiments

## 1. ImageNet classification

### 학습 데이터 셋

- 1,000개 클래스 + 130만 개 훈련 이미지 + 50,000개의 테스트 이미지

### Data Augmentation

- 크기(size) 및 가로 세로(aspect) 비율 왜곡
- 랜덤 크롭(random crop)
- 수평 플립(horizontal flips)

### learning rate

- 처음 5 epochs 동안)0.1 → 0.4로 선형적으로(linearly) 증가
- 90 epochs 동안 training 시) 30, 60, 80 epochs 이후에 0.1배
- 200 epochs 동안 training 시) 60, 120, 180 epochs 이후에 0.1배

### 테스트 데이터 셋

- 224 x 224의 크기로 중간 부분이 소실된 이미지로 테스트

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0d90841e-99eb-4486-8b9d-73c95c137de7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104621Z&X-Amz-Expires=86400&X-Amz-Signature=00f53bd648f813925b41581e3dd10ddb61428bd0355a737cd63406b33738a8cf&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### 실험 내용

- mixup과 ERM을 사용 → SOTA ImageNet-2012 분류 모델을 학습

### 실험 결과

- hyperparameter α

→ 0.1~0.4에서 우수한 성능 + 이보다 더 클 경우, underfitting

- 90 epochs) ResNet-101과 ResNeXt-101 모델은 ResNet-50 소형 모델(0.2%)보다 훨씬 개선(0.5% ~ 0.6%)됨
- 200 epochs) mixup은 ResNet-50의 top-1 error가 90 epochs에 비해 1.2% 더 감소하지만, ERM은 ResNet-50의 top-1 error가 90 epochs와 비슷하게 유지

⇒ ERM과 비교했을 때, mixup은 **higer capacities**와 longer training run의 효과 

📎 higer capacities

→ higher capacity (= parameter 개수 = hidden layer 개수) ⇒ 더 high-level feature를 추출할 수 있게 된다

→ 구체적으로 hidden layer가 쌓이는 과정에서 비선형 함수를 사용하며, 함수들이 합성되면서 훨씬 더 복잡한 함수를 표현 가능 ⇒ 더 high-level feature 학습 가능 ⇒ 더 어려운 문제를 해결 가능 ⇒ representive(표현력)이 더 좋아짐

## 2. CIFAR-10 and CIFAR-100

- mixup의 generalization performance를 평가하기 위해서 실험

### setting

- 200 epochs + 128 minibatch

### learning rate

- 0.1에서 시작
- 100, 150 epochs에서 0.1배 (단, WideResNet은 제외)
- WideResNet은 60, 120, 180 epochs에서 0.1배

### Figure 3 (a)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/dc180f4a-fef5-4430-8449-1812f5c5fda4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104640Z&X-Amz-Expires=86400&X-Amz-Signature=0f9b0dd263269e36bf9647bd824993511ff637b80c162010f0595069ac762a6a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

- 성능 : ERM < mixup

### Figure 3 (b)

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/cad7cafd-7c58-44dd-bc27-7014ffdb73f8/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104653Z&X-Amz-Expires=86400&X-Amz-Signature=db135c0f8bc6fa3575f2f8f113fdb5ab8e12653602612eafa8f627379b7825b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

- ERM와 mixup은 비슷한 속도로 각각의 best test error로 수렴

## 3. Speech Data

- 음성 인식 실험을 수행

### 데이터 셋

- Google commands dataset
- 65,000개의 발화를 포함
- 각 발화는 약 1초 + 30개 클래스(수천 명의 사람이 발음하는 yes/no/down/left 같은 음성 명령)

### 전처리

- 원래 파형에서 16kHz의 샘플링 속도로 정규화된 spectrogram을 추출
- 160 × 101로 크기를 동일하게 하기 위해 spectrogram을 zero-padding
- 음성 데이터의 경우 파형과 spectogram에서 mixup을 적용하는 것이 좋으나, 여기서는 spectogram에 mixup을 적용

### setting

- 30 epochs + 100 minibatch

### learning rate

- ![image](https://latex.codecogs.com/gif.latex?3%5Ctimes%2010%5E%7B-3%7D)에서 시작
- 10 epochs 마다 0.1배

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1f04cf6d-7c73-47ec-ba1b-77ca97169685/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104719Z&X-Amz-Expires=86400&X-Amz-Signature=3c95fe2f34a3bf422a6e9f0f7464fe2373e8752cf224e9e77dde045dbbf77ef3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### 실험 결과

- 용량이 더 큰 모델인 VGG-11에서 mixup이 ERM을 능가

## 4. Memorization of Corrupted Labels

- memorization을 검증하기 위해서 손상된(corruption) label로 학습

### 가정

- hyperparameter α가 커질수록 memorization이 어려워진다

(∵ α가 커질수록 virtual example을 training data로부터 멀리에서 생성하기 때문에)

### 데이터 셋

- 각 레이블의 20%, 50%, 80%에 랜덤 노이즈 발생
- 모든 테스트 라벨은 그대로 유지

### 비교할 모델

- Dropout → 손상된 label을 사용한 학습에 대해 SOTA 모델
- mixup / Dropout / mixup+Dropout / ERM을 비교

### setting

- mixup) α ∈ {1, 2, 8, 32}을 선택
- Dropout)  p ∈ {0.5, 0.7, 0.8, 0.9}을 선택
- mixup+Dropout) α ∈ {1, 2, 4, 8}과 p ∈ {0.3, 0.5, 0.7}을 선택

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b7a7a792-3b3d-4320-9186-763d268663e0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104741Z&X-Amz-Expires=86400&X-Amz-Signature=f2223f84376c88b7e941a64d7164754cb253658baa4677791f564a93ebf169ad&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### 표

- best error와 200 epochs 후의 error를 기록
- memorization 정도를 정량화하기 위해서 실제 label과 손상된 label에 대한 마지막 epochs에서의 error도 평가

### 실험 결과

- 작은 learning rate로 진행되면서, ERM 모델은 손상된 label에 과적합하기 시작함
- 큰 확률 p(ex. 0.7, 0.8)를 사용하는 경우, Dropout으로 과적합을 줄일 수 있음
- 큰 α(ex. 8, 32)을 사용하는 경우, mixup은 best/last error 모두 Dropout보다 성능이 좋음
- mixup+Dropout이 가장 좋은 성능을 보임

## 5. Robustness to Adversarial Examples

### Adversarial Example

→ 모델의 성능을 저하시키기 위해, 틀린 클래스로 인식하도록 작은 노이즈를 냄 (단, 노이즈를 포함한 사진이 사람이 보기에는 원래 사진과 구분되지 않아야 함)

### setting

- **FGSM(Fast Gradient Sign Method)** / Iterative-FGSM(I-FGSM) 방법을 사용 → Adversarial Example 생성
- I-FGSM의 경우 동일한 step 크기로 10회 반복
- 모든 픽셀에 대해 최대 4까지의 노이즈(perbutations)를 허용

📎 FGSM

입력 이미지에 대한 **gradient(기울기) 정보**를 추출하고, 이를 왜곡하여 원본 이미지에 더하는 방법

반복된 학습 X + 공격 목표를 정할 수 X(= **non-targeted** 방식) + white box attack 

📎 gradient(기울기) 

모델이 학습할 때 각 픽셀이 미치는 영향

📎 targeted vs. non-targeted

targeted → 원하는 정답으로 유도 가능

non-targeted → 원하는 정답으로 유도 불가능

📎 one-shot vs. iterative

one-shot → 잡음을 생성하기 위한 반복된 학습(최적화)가 필요 X (ex. FGSM)

iterative → 잡음을 생성하기 위한 반복된 학습(최적화)가 필요 O (ex. I-FGSM)

📎 White-box attack 

모델의 정보(ex. gradient)를 토대로 노이즈 생성 

📎 Black-box attack 

모델 정보 없이 노이즈 생성 

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e65d594b-92d8-430b-babc-f3bc3dd7d96a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104804Z&X-Amz-Expires=86400&X-Amz-Signature=b864032c36f2a11fe0572cdf4dad062edd5f5bf3bda2928528b62fa05d0d2519&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### 실험 결과

- (a) FGSM의 경우, mixup 모델이 Top-1 error에서 ERM 모델보다 2.7배 더 강력하다
- (b) FGSM의 경우 mixup 모델이 Top-1 error에서 ERM 모델보다 1.25배 더 강력하다
- (a) mixup과 ERM 모두 I-FGSM에 강하지 않지만, (b) I-FGSM에서 mixup은 ERM보다 40% 더 강력하다

ex. 2.7배 강력하다 = (100 - Top-1 mixup error(= 75.2)) / (100 - Top-1 ERM error(= 90.7) ) = 2.7

- white box attack에 대해 강력한 정도 : ERM < mixup
- black box attack에 대해 강력한 정도 : ERM < mixup

## 6. Tabular Data

- 비이미지 데이터에 대한 mixup의 성능을 추가로 탐구하기 위해 6가지 임의 분류 문제 실험

### setting

- minibatch size 16 + 10 이상의 epochs

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/aa8234b6-340e-4675-8478-f5734a776c6a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104820Z&X-Amz-Expires=86400&X-Amz-Signature=0d22dacaaed8b9db52f7a776b39364334389e9206bbad0512d895848544ff2a1&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### 실험 결과

- 성능 : ERM < mixup

## 7. Stabilization of Generative Adversarial Networks

### GAN의 이미지 생성 원리

- GAN에서 생성자(generator)와 판별자(discriminator)는 분포 P를 모델링하기 위해 **경쟁**한다
- 생성자(g)는 노이즈 벡터 z를 실제 샘플 x ~ P와 유사한 가짜 샘플 g(z)로 변환하기 위해 경쟁
- 판별자(d)는 실제 표본 x와 가짜 표본 g(z)를 구별하기 위해 경쟁
- 수학적으로, GAN을 훈련시키는 것은 최적화 문제를 해결하는 것과 같다

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7f3f1c1b-abc8-48ec-8aa7-e25900df7378/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104837Z&X-Amz-Expires=86400&X-Amz-Signature=9ff9fcc88f530f6d30c1288ca3343559f7a8b51dca0fec24130eff7f0ac33e0f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

- ![image](https://latex.codecogs.com/gif.latex?l)은 **이진 교차 엔트로피 손실(binary cross entropy loss)**
- 판별자는 생성자에 vanishing gradient 문제를 일으키므로, min-max 방정식을 해결하는 것은 매우 어려움

📎 이진 교차 엔트로피 손실(binary cross entropy loss)

이진 분류(= 데이터가 주어졌을 때, 해당 데이터를 두 가지 정답 중 하나로 분류하는 것)의 목적 함수로 사용

📎 “경찰과 위조지폐범” 비유

위조지폐범(generator)은 위조지폐를 진짜 지폐와 거의 비슷하게 만듭니다. 경찰(discriminator)은 이 지폐가 진짜 지폐인지 위조지폐인지 구분하려 합니다. 위조지폐범과 경찰은 적대적인 관계에 있고, 위조지폐범은 계속 위조지폐를 생성하고 경찰은 진짜를 찾아내려고 하는 쫓고 쫓기는 과정이 반복됩니다. 

⇒ 위조지폐범(Generator)와 경찰(Discriminator)가 서로 위조지폐를 생성하고 구분하는 것을 반복하는 minmax game

### mixup 모델이 GAN 훈련을 안정화

- mixup이 판별자의 gradient 정규화(regularizer) 역할 (cf. Figure 1 (b)) → GAN의 훈련을 안정화
- 판별자의 **smoothness(부드러움)은 생성자에 안정적인 gradient 정보를 보장** → **vanishing gradient** 문제 해결

📎 smoothness가 안정적인 gradient 정보를 보장?

(생각) ERM처럼 클래스 1/클래스 0으로 뚜렷하게 나뉘면 오차가 급격하게 줄어드는데에 반해, smoothness를 가진 mixup의 경우, 오차가 급격하게 줄어들지 않으므로 vanishing gradient 문제를 해결할 수 있다 

📎 vanishing gradient

오차가 크게 줄어들어 학습(ex. 가중치 업데이트)이 되지 않는 현상 

### mixup을 적용한 GANs

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bb8fe95a-80b6-42d3-b62b-7f2f23b3adc5/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104922Z&X-Amz-Expires=86400&X-Amz-Signature=885a75b052fa410d6a16913ad8cd86a74c96b86069eac2a5b089fbb2fa303a8d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### Figure 5

<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/dd19f005-2b6f-4fe3-a155-d9e2a011435b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220127%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220127T104937Z&X-Amz-Expires=86400&X-Amz-Signature=57d55a0ed3ce279e003b4ef75498519dd2cfff3a1ae90997856c8b31ef88f738&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="50%" height="50%">

### setting

- 120,000 minibatch + minibatch size 128
- 모든 생성자의 훈련을 시작하기 전에 판별자가 5회 반복해서 훈련된다

### 실험 결과

- 두 개의 데이터 셋(파란색 샘플)을 모델링할 때, GAN(주황색 샘플)의 훈련을 안정화된 정도 : ERM < mixup

### 참고자료

- [선형결합?](https://m.blog.naver.com/cindyvelyn/221855297366)
- [convex combination? (1)](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nywoo19&logNo=221608580563)
- [convex combination? (2)](https://light-tree.tistory.com/176)
- [convex combination? (3)](https://en.wikipedia.org/wiki/Convex_combination)
- [SOTA?](https://joft.site/106#:~:text=SOTA%EB%8A%94%20State%2Dof%2Dthe,%EC%9D%98%20%EA%B2%B0%EA%B3%BC%EB%A5%BC%20%EC%9D%98%EB%AF%B8%ED%95%9C%EB%8B%A4.&text=kaggle%EC%97%90%EC%84%9C%20%EB%AA%A8%EB%8D%B8%20%EA%B5%AC%EC%B6%95%EC%9D%84,%EC%9D%98%20%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%B4%EB%9D%BC%EB%8A%94%20%EB%9C%BB%EC%9D%B4%EB%8B%A4.)
- [mixup 논문? (1)](http://dmqm.korea.ac.kr/activity/seminar/307)
- [mixup 논문? (2)](https://techy8855.tistory.com/19)
- [mixup 논문? (3)](https://everyday-image-processing.tistory.com/145)
- [mixup 논문? (4)](https://deepseow.tistory.com/17)
- [mixup 논문? (5)](https://www.notion.so/mixup-467e0a5d4d284e05a5879007b9d1b97f)
- [Data Augmentation?](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0015530517&dbt=DIKO#:~:text=%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A6%9D%EA%B0%95(Data%20Augmentation)%EC%9D%80,%EC%9D%B4%20%EB%B6%80%EC%A1%B1%ED%95%9C%20%EA%B2%BD%EC%9A%B0%EA%B0%80%20%EB%A7%8E%EB%8B%A4.)
- [선형 보간법?](https://sdc-james.gitbook.io/onebook/4.-numpy-and-scipy/4.3-scipy/4.3.1.-interpolation)
- [경험적 분포?](https://ko.wikipedia.org/wiki/%EA%B2%BD%ED%97%98%EC%A0%81_%EB%88%84%EC%A0%81_%EB%B6%84%ED%8F%AC_%ED%95%A8%EC%88%98)
- [베타 분포?](https://soohee410.github.io/beta_dist)
- [regularization?](https://wikidocs.net/120052)
- [gradient norm?](https://aimaster.tistory.com/95)
- [higher capacities? (1)](https://etyoungsu.tistory.com/9)
- [higher capacities? (2)](https://warm-uk.tistory.com/53)
- [white box attack vs. black box attack?](https://lepoeme20.github.io/archive/FGSM)
- [FGSM? (1)](https://velog.io/@seoyeon/Pytorch-tutorial-%EC%A0%81%EB%8C%80%EC%A0%81-%EC%98%88%EC%A0%9C-%EC%83%9D%EC%84%B1ADVERSARIAL-EXAMPLE-GENERATION)
- [FGSM? (2)](https://yjs-program.tistory.com/171)
- [GAN의 이미지 생성 원리?](https://ysbsb.github.io/gan/2020/06/17/GAN-newbie-guide.html#:~:text=GAN%20(Generative%20Adversarial%20Network)%EC%9D%80,%EC%97%90%20%EB%84%90%EB%A6%AC%20%EC%93%B0%EC%9D%B4%EB%8A%94%20%EB%AA%A8%EB%8D%B8%EC%9E%85%EB%8B%88%EB%8B%A4.&text=%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A5%BC%20%EC%83%9D%EC%84%B1%ED%95%98%EB%8A%94%20%EC%9B%90%EB%A6%AC,%EC%A7%80%ED%8F%90%EC%99%80%20%EA%B1%B0%EC%9D%98%20%EB%B9%84%EC%8A%B7%ED%95%98%EA%B2%8C%20%EB%A7%8C%EB%93%AD%EB%8B%88%EB%8B%A4.)
- [이진 교차 엔트로피?](https://wooono.tistory.com/387)
