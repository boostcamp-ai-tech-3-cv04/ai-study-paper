## 요약
- **Convoultion network** : 최신 컴퓨터 비전 분야의 여러 다양한 분야에서 사용되는 제일 핵심 기술

- (가정 : 훈련을 위한 충분히 라벨링된 데이터가 주어졌다) 모델 크기, 계산량 증가 → 품질 향상  
  **But** 모바일 비전이나 빅데이터 분야에서 `계산 효율성`과 `적은 매개변수`는 중요한 부분

- 실험 목적
  1. Suitably factorized convolutions
  2. Aggressive regularization
  
  → 1,2를 통한 네트워크의 크기를 효율적으로 키우는 방법 탐색

## 일반적인 설계 원칙

언급되는 원칙들은 추측성으로 정확성이나 검증에 대해서는 실험적인 증거가 필요하지만 이 원칙들을 벗어나기만 해도 네트워크의 성능이 떨어졌으며, 해당 벗어난 부분을 고치기만 하면 아키텍처가 일반적으로 개선되는 경향을 보였다.

### 1. **Representational Bottleneck을 피해라.**
  
Feed-forward(전방향 공급) 네트워크의 경우 입력 계층에서 classifier이나 regressor 간의 비순확적인 그래프 표현이 가능히다. → 정보의 흐름에 명확한 방향 존재  

Input에서 Output까지의 모든 layer의 경우, 각 layer를 통과하는 정보량에 접근할 수 있다. 이 때, 극단적인 압축으로 인한 **bottleneck**을 피해야한다.  

> 특히 초반부에서 일어나는 bottleneck을 주의해야한다. 초반에서 bottleneck이 일어나서 정보 손실이 발생한다면 아무리 네트워크가 깊어져도 정보의 원천이 부실해지므로 성능의 한계가 발생하기 때문이다.
  
일반적으로 input에서 final representation까지 도달하면서 representation size가 서서히 감소해야한다.

> Representation : 각 layer의 출력

이론적으로 correlation structure와 같은 중요한 요소를 버리기 때문에, 정보를 representation의 차원으로만 평가할 수 없다.

### 2. **고차원 Representation은 네트워크 내에서 지역적으로 처리하기 쉽다.**  

CNN에서 activation per tile을 증가시키면 disentangle feature들을 많이 얻을 수 있어 네트워크가 더 빠르게 학습하게 된다. 
  
### 3. spatial aggregation은 representational power를 그렇게 많이 잃지 않으면서도 저차원 임베딩에 적용될 수 있다.  

더 많은 Convolution을 수행하기 전에, 심각한 부작용 없이 input representation의 차원축소가 가능하므로, 그 후에 spatial aggregation을 할 수 있다. 또한 이러한 signal은 쉽게 압축할 수 있다는 점을 생각하면 차원축소로 인해 학습 속도가 빨라질 것이다.

> 여기서의 spatial aggregation은 convolution 연산을 표현하는 것으로 보인다.

> signal 압축이란, 학습 과정에서 네트워크의 각 layer를 거쳐가며 원하는 동작을 위한 판단에 필요한 feature를 입력 데이터로부터 추출하는 작업이다. 즉, 많은 convolution을 수행하는 경우 적절한 차원 축소를 해주는 것이 빠른 학습에 도움이 된다는 것이다.

이는 출력이 spatial aggregation에 사용되는 경우, 인접한 unit 간의 강력한 상관 관계로 인해 dimension reduction에서의 정보 손실이 훨씬 줄어들 것이라는 가설에 근거한 원칙이다.

### 4. 네트워크 width와 depth의 균형을 맞춰야 한다
네트워크의 최적의 성능은 각 단계별 필터의 개수와 네트워크의 depth간의 balance로 결정된다.  
네트워크의 width와 depth가 증가하면 더 높은 수준의 네트워크를 만들 수 있다. 이 때, 둘을 병렬적으로 증가시킨다면 일정한 계산량에 대한 optimal improvement에 도달할 수 있다.  
단, compuational budget이 width와 depth 간의 균형 잡힌 방식으로 할당되도록 네트워크를 구성해야 최적의 성능을 보인다.

> width는 각 stage의 filter의 개수이다.

## 합성곱 분해

GoogLeNet은 1x1 convolution로 차원을 축소시키고 3x3 convolution을 여러개 활용하여 파라미터 수를 감소시켰다. 이처럼 합성곱을 분해하는 것은 파라미터수를 감소시킬 수 있다. 파라미터 수가 감소하면 연산량도 감소하여 빠른 학습이 가능하다.

### (1) 더 작은 합성곱으로 분해

큰 spatial filter를 갖는 convolution은 계산적인 측면에서 불균형하게 비싼 경향이 있다. 

> n개의 filter로 이루어진 5x5 convolution 연산의 경우, 같은 수의 filter를 사용하는 3x3 convolution보다 25/9배 이상으로 비싸다.

5x5 convolution을 매개변수는 더 적으면서도 같은 입력 크기 및 출력 심도를 갖는 다중계충 네트워크로 대체할 수 있을까

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152102442-a0243a23-6b3f-4aeb-9dda-2cb6b3a5f7c9.png" width=300></p>

여기선 vision network를 구축하고 있기 때문에, fully-connected component를 2-layer convolution로 대체하여 translation invariance을 다시 이용하는 것이 자연스러워 보인다.

> Translation invariance는 입력에 shift가 일어난 경우에도 변함 없이 학습한 패턴을 캡처하는 convolution 방식의 특성을 말하는 것이다. 

즉, 위 그림처럼 5x5 convolution은 3x3 convolution 2개로 분해할 수 있다. 5x5 convolution은 25번의 연산을 하고, 3x3 convolution 2번은 각각 9번의 연산을 수행한다. 따라서 연산량을 18/25배가 된다. 파라미터 수 또한 18/25배가 된다. 28%의 이득을 얻을 수 있다.

> 질문 1. 위와 같은 대체로부터 표현력 손실이 발생하는가?
> 
> 질문 2. 계산의 linear part에 대한 factorizing이 목적인 경우에, 2-layer 중 첫번째 layer에서는 linear activation을 유지해야하는가?

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152105657-1f9e3ac1-26c0-4f74-ac42-c121f9aca4b2.png" width=500></p>

- red line : activation으로 linear과 ReLU 사용
- blue line : 두 activation 모두 ReLU 사용

실험을 통해, factorization에 linear activation을 사용하는 것이 모든 단계에서 ReLU를 사용하는 것보다 성능이 좋지 않다는 것을 확인할 수 있다. 또한 Batch normalization을 사용하면 정확도가 높아지게 된다.

**따라서 3x3 convolution보다 큰 fiter는 언제든지 3x3 convolution으로 분해하여 사용하는 것이 좋다.**

실제로 Inception module에서 5x5 convolution 부분을 3x3 convolution으로 대체한다.

<img src="https://user-images.githubusercontent.com/57162812/152106300-25896757-0384-4300-b147-7a154adc3978.png" width=300><img src="https://user-images.githubusercontent.com/57162812/152106366-81214336-e28a-4010-8ee4-2a472450c31d.png" width=300>

### (2) 비대칭 합성곱(Asymmetric Convolution) 분해

그렇다면 3x3 convolution을 더 작은 convolution으로 분해할 수 있을까? 예를 들면 2x2 convolution으로 축소할 수 있다. 하지만, 2x2보다는 nx1과 같은 비대칭 convolution을 사용했을 때 더 성능이 좋다.

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152107005-25efde80-c3c7-487a-9f3e-82759363c47e.png" width=500></p>

3x1 convolution 뒤에 1x3 convolution을 사용한 2-layer를 sliding하는 것과 3x3 convolution의 receptive field는 동일하다.

입출력의 filter 수가 같은 경우에는, 같은 수의 output filter에 대해서 2-layer solution이 (3+3)/9=0.66배로 계산량이 감소된다.

반면에 2x2로 분해하면 (4+4)/9=0.89배로 절약된다.

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152107541-12b9640f-836b-4b92-98b3-3b80f5645dc4.png" width=200></p>

위는 nxn convolution을 factorizing한 inception module이다.

## 보조 분류기의 활용

GoogleNet은 very deep network의 수렴을 개선시키기 위해 보조 분류기(Auxiliary Classifier)를 도입했다.

보조 분류기의 원래 동기는 다음과 같다.
- Useful한 gradient를 하위 layer로 밀어 넣어, 즉시 useful하게 만들기 위함
- Very deep network의 기울기 소실 문제를 해결하여, 학습 중의 수렴을 개선시키기 위함

하지만, 학습 초기에는 보조 classifier들이 수렴을 개선시키지 않는다는 흥미로운 결과를 발견했다고 한다.

또한, GoogleNet에서는 두 개의 보조 분류기가 각각 다른 stage에 사용되었지만, 하위 stage의 보조 분류기 하나를 제거하더라도 최종 성능에 악영향을 미치지 않았다고 한다. 

따라서 원래의 GoogleNEt에서 세운 가설인 "보조 분류기가 low-level feature의 발전에 도움이 된다"는 것이 잘못된 것일 도 있음을 알 수 있다. 그 대신, 이번 연구에서는 "이러한 보조 분류기가 regularizer로 동작한다."고 주장한다.

> 이는 보조 분류기에서 BN이나 drop out이 사용되는 경우, 주 분류기의 결과가 더 좋다는 사실이 근거가 되는 주장이다.

## 효율적인 그리드 크기 축소

CNN은 pooling 연산을 통해서 feature map의 grid size를 줄인다. 이 때, representational bottleneck을 피하기 위해, pooling을 적용하기 전에 activated filter의 차원이 확장된다.

> 예를 들어, dxd grid에 k개의 filter로부터 시작해서, d/2 x d/2 grid와 2k개의 filter에 도달하려면, 먼저 2k개의 filter로 stride가 1인 convolution을 계산한 후에 pooling을 수행한다.

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152113688-e3bb8cc2-6cdd-4ee4-a8f7-ff28e40f5039.png" width=400></p>

[Fig9 오른쪽] 네트워크 전체 계산 비용이 pooling 이전의 확장 단계에서 일어나는 2d^2k에 좌우한다는 것을 의미한다. 따라서 비용이 비싸진다.

[Fig9 왼쪽]만약 convolution과 pooling의 순서를 바꾼다면, 계산 비용이 1/4로 줄어 2(d/2)^2k가 된다. 하지만 이는 representation의 전반적인 차원이 (d/2)^2xk로 낮아져서 표현령ㄱ이 떠러지게 되고, 이는 곧 representational bottlenek을 야기한다.

➡ representational bottleneck을 피하면서 계산 비용도 줄일 수 있는 구조 제안
: stride가 2인 block2개를 병렬로 사용
- 블록 = pooling layer+conv layer
- pooling : maximum 혹은 average
- 두 블록은 concatenate로 연결

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152114116-ce2e9fd6-85c7-409a-a138-ca620c2e608e.png" width=400></p>

> 좌측 다이어그램에 따르면, convolution part는 두 개의 branch로 이뤄져있음을 알 수 있다. 여기서 두 branch 간의 filter 수의 비율은 언급되지 않았지만, 절반으로 가정하고 계산해보자. 우선 2-layer인 branch에서는 stride가 1인 것과 2인 conv layer에서 각각 d^2k와 (d/2)^2k만큼 비용이 발생하며, 1-layer인 branch에서는 (d/2)^2k만큼의 비용이 발생한다. 따라서 총 3/2d^2k만큼의 비용이 발생한다. 기존의 2d^2k에 비하면 비용이 25% 저렴하다고 할 수 있다.

## Inception v2

지금까지 설명한 개념을 결합한 Inception v2의 네트워크 구조 개요이다.

<p align="center"><img src="https://user-images.githubusercontent.com/57162812/152115370-c19e4424-9ea8-4d4f-82af-250e69a2e2da.png" width=350></p>

Inception module에서 conv 연산은 feature map의 크기를 유지하기 위해 0-padding 적용하였고 그 이외에는 0-padding을 적용하지 않았다.

Inception-v2에는 3종류의 inception module을 사용한다.

<img src="https://user-images.githubusercontent.com/57162812/152116423-817a6047-5469-425e-97da-c7d038f280dc.png" width=200><img src="https://user-images.githubusercontent.com/57162812/152116604-e2b3e333-c7a0-463c-98c8-63b8c524ba33.png" width=200><img src="https://user-images.githubusercontent.com/57162812/152116857-96dc699c-dd00-4631-bf65-ef2313ab332a.png" width=200>

## Model Regularization via Label Smoothing

Label Smoothing은 one-hot으로 인코딩 된 label vector uniform distribution의 결합이다.

*new_labels=(1-ε)\*one_hot_labels+ε/K*

- K : 클래스의 개수
- ε : 하이퍼 파라미터

만약 ε=1이면 uniform distribution이고 ε=0이면, 그냥 encoding된 one-hot vector이다.

예를 들어 개/고양이 이진분류에서 ε=0.2로 설정되었다고 해보자. 그리고 0은 고양이, 1은 개라는 rule로 labeling을 했다고 하자. 위의 식을 이용하면 새로운 label은 아래와 같이 계산된다.

new_label=1\*(1-0.2)+0.2/2=0.9

그리고 원래 0으로 부여되었던 label의 경우 1-0.9=0.1로 부여하게 된다. 따라서 모델이 target을 정확하게 예측하지 않아도 되도록 만드는 것이다. 원래 1인 데이터에 대해서 모델은 1을 최대한 예측하도록 트레이닝 되는데 label smoothing을 적용하면 0.9 정도면 잘 예측했다 정도의 loss 값을 산출하게 되는 것이다. 그렇기 때문에 모델이 overconfident 해지는 경향을 막게 되어 regularization이 되는 효과를 갖게 된다.

## 낮은 해상도/작은 크기의 이미지에서의 성능(Performance on Lower Resolution Input)

일반적으로, higher resolution의 receptive field를 사용하는 모델이, recognition 성능이 크게 향상되는 경향이 있다고 알려져 있다.

> CNN에서 Receptive field는 각 단계의 입력 이미지에 대해 하나의 필터가 커버할 수 있는 이미지 영역의 일부를 뜻한다.
d
**layer의 receptive field의 resolution이 증가했을 때의 효과와, model이 커짐에 따른 capacitance 및 computation에 대한 효과를 구별하는 것이 중요하다.**

> Receprive field의 resolution이 커진다는 것은 convolution filter와 input 간의 weight sum의 계산에 사용되는 pixel의 수가 많아진다는 것이다. Resolution이 커질수록 더 넓은 범위의 인근 pixel들을 고려하여 패턴을 학습할 수 있게 된다.
>
> 모델의 capacitance가 크다는 것은 많은 parameter를 가지는 것이며, 그만큼 더 복잡한 관계에 대해 패턴을 학습할 수 있는 여지가 생긴다. 예를 들어, 3-layer를 가지는 CNN으로 MNIST dataset에 대한 학습을 진행하면 우수한 성능을 얻을 수 있지만, ImageNet dataset에 대한 학습을 진행하면 좋지 못한 성능을 얻게 되는 것과 유사한 이치이다.

만약 모델을 수정하지 않고 input resolution만 변경한다면, 계산 비용이 훨씬 저렴한 모델로, 보다 어려운 작업에 대한 학습을 하게 된다. 이 경우, 계산량이 줄어드는만큼 솔루션의 견고함도 떨어지게 된다.

일정한 계산량을 유지하는 간단한 방법은, 입력이 lower resolution인 경우에 처음 두 layer에서 stride를 줄이거나, 네트워크의 첫번째 pooling layer를 제거하면 된다. 이를 위해 다음 세 가지 실험을 수행했다.

1. stride가 2인 299x299 receptive field를 사용하고, 첫번째 layer 다음에 max pooling을 사용
2. stride가 1인 151x151 receptive field를 사용하고, 첫번째 layer 다음에 max pooling을 사용
3. stride가 1인 79x79 receptive field를 사용하고, 첫번째 layer 다음에 pooling layer가 없음

각각의 네트워크는 수렴 될 때까지 학습했으며, 성능은 ImageNet ILSVRC 2012 classification benchmark의 validation set에 대해 측정됐다.

<img src="https://user-images.githubusercontent.com/57162812/152136317-1862bdb6-8a74-4921-8232-a5b959de9555.png" width=400>

lower-resolution 네트워크가 학습하는 데 오래 걸리긴 하지만, 성능은 higher resolution 네트워크에 근접한다.

## Experimental Result and Comparision
#### 다양한 기법들에 대한 누적 효과를 비교하는 single-crop 성능
![image](https://user-images.githubusercontent.com/57162812/152137928-cedda2fc-c342-4b80-ad6f-5f40a8f0e39d.png)

#### Single-model, Multi-crop 실험 결과
![image](https://user-images.githubusercontent.com/57162812/152138010-9ab1c541-48db-4ba1-90f8-50ca71995b09.png)

Inception-v3=Inception-v2+RMSProp+Lable Smoothing+ Factorized 7x7+BN-auxiliary

#### Multi-model, Multi-crop 실험 결과
![image](https://user-images.githubusercontent.com/57162812/152138072-62e118cf-0a71-4626-935c-c35bc76deda2.png)
