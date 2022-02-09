# 목차

- [**Abstract**](#1-abstract)
- [**Introduction**](#2-introduction)
- [**Related work**](#3-related-work)
- [**EfficientNet 구조**](#4-efficientnet-구조)
- [**결과**](#5-결과)

# 1. Abstract

**이 논문에서는 컴퓨터 자원의 한계, 여유에 맞춰서 network의 depth, width, resolution을  scale up 하는 방법에 대한 정량적인 해법과 그에 맞는 새로운 network를 제시하였다.** 

- CNN은 제한된 resource 내에서 발전해왔고, 만약 더 많은 resource 제공이 된다면 scale up을 통해서 정확도를 높여왔다. 

- 기존에는 scale up 방식으로 width, depth, resolution 중 하나만을 늘리는 방법을 택했다면, 이 논문에서는 이 세가지 방식을 모두 사용하는 것에 대한 실험적인 결과를 내놓았다. 

  >  이 세가지를 균등하게 (compound coefficient) 증가시키는 방법을 제시하며, 이러한 방식을 적용할 baseline model까지 design했다.  

-  특히, EfficientNet-B7은 SOTA CNN 모델에 비해 8.4배 이상 작고, 6.1배 이상 빠르다.



# 2. Introduction

모델이 scaled up 될수록 성능이 좋아지는 경우가 많다.

ex) ResNet-18 -> ResNet-200, GPipe -> 4 x baseline model

기존 모델들에서는 width, depth, resolution 중 하나만을 택해 scale up 하였다.

![image-20220209141925676](../AppData/Roaming/Typora/typora-user-images/image-20220209141925676.png)

위 사진은 **기존 모델 scale up 방식 (a, b, c, d)** 과 **논문에서 발표한 방식 (e)** 을 비교한 사진이다. 

![image-20220209142321032](../AppData/Roaming/Typora/typora-user-images/image-20220209142321032.png)

이전 논문들에서 이론, 실험적으로 width와 depth의 관계를 증명했지만, 

이 논문에서는 정량적으로 **세 차원의 balance를 조절하면서 scale up**하는 것이 중요하다는 것을 보여주었고, **고정된 coefficient**로 세 차원을 모두 scale up하는  **Compound Scaling Method**를 소개하였다. 

> 세 차원을 balance 있게 scale up 하는 것은 직관적으로 생각해보면 당연하다. 
>
> resolution, 즉 input 이미지의 크기가 증가하면 receptive field를 증가시키기 위해서 layer를 증가시켜야 한다. 
>
> 또한, 이미지에서 더 섬세한 pattern을 뽑아내기 위해 더 많은 채널이 필요하게 된다. 
>
> (여기서 receptive field는 ouput의 pixel이 나오기 위해서 영향을 미치는 input pixel들이라고 생각하면 되는데, 이것을 가볍게 kernel이라고 생각한다면, 만약 동일한 3x3의 커널이 더 넓은 영역에서의 pattern을 뽑아내기 위해서는 layer를 증가시켜, 이 kernel을 여러번 적용하면 된다. )

하지만, model scaling은 baseline network에 dependent하기 때문에 neural architecture search [1]를 통해 새로운 baseline network를 개발했고, 이 model을 scale up해서 EfficientNet을 만들었다. 

![image-20220209142630777](../AppData/Roaming/Typora/typora-user-images/image-20220209142630777.png)



# 3. Related Work

hardware 메모리 한계에 도달했기 때문에, 더 좋은 accuracy향상에는 더 좋은 efficiency가 필요하며,  model compression[2] 은 model 사이즈를 network의 width, depth, kernel size와 kernel type 변경함으로 모델 크기를 축소시켰다. 

(물론 accuracy는 기존 모델에 비해 조금 떨어진다.)

최근 들어서는 이러한 연구가 활발해졌고, 오히려 모델 축소가 mobile ConvNet보다 더 효율적인 결과를 보였다. 

### 3-1. Model scaling

원래 ConvNet의 식을 아래로 표현할 수 있다. 

(F는 operator, X는 input, N은 ConvNet)

![image-20220131214754153](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131214754153.png)

어차피 같은 block안에서 layer는 반복되는 것이 대부분이므로

 (resnet도 처음 down sampling빼면 같은 구조의 layer가 반복된다.) 

아래 식으로 표현 가능하다.  

![image-20220131214640895](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131214640895.png)

(L은 i번째 stage에서 Fi가 Li번 반복된다는 의미)

각 layer마다 Li, Ci, Hi, Wi를 서로 다르게 바꾸려면 design space가 늘어나기 때문에 이 논문에서는 모든 layer에 대해서 변화를 **상수 비율로 균등**하게 맞췄다. 

(scale up의 목적은 Acc의 향상과 효율이므로 아래 식으로 표현 가능, r, w, d는 각각의 constant ratio)

![image-20220131215523692](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131215523692.png)

### 3-2. 각각의 dimension scale up의 효과

- **Depth**

  - 효과
    - capture richer, complex features
    - generalization performance 향상

  - 단점
    - vanishing gradient (skip connection이나 BN에도 불구하고 존재하는 문제)


- **width**

  - 특징
    - width는 scale up은 보통 작은 network에서 사용됨

  - 효과
    - fine-grained feature를 더 잘 뽑아내고 train이 쉽다. 

  - 단점
    - 너무 넓고 얕은 network는 higher level feature를 뽑아내기 어렵다. 


- **Resolution**
  - 효과
    - potentially fine-grained pattern을 잘 뽑아낸다. 


![image-20220209145022920](../AppData/Roaming/Typora/typora-user-images/image-20220209145022920.png)

위의 사진의 결과로 알 수 있듯이, **각각의 dimestion을 늘리는 건 accuracy향상에 도움이 되지만, 일정 수준(80%)에 도달한 모델들에서는 accuracy gain이 사라진다.** 



### 3-3. Compound scaling (3dimesion 변화)

![image-20220131223404973](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131223404973.png)

위 사진은 w (width)를 증가시켰을 때, d와 r의 값에 따른 정확도를 표현한 그래프이다. 

위의 결과로 미루어보아, w만 증가시켰을 때보다, d와 r을 같이 증가시켰을 때 훨씬 좋은 성능을 보였다. 

**따라서, w, d, r을 모두 balance 있게 scale up하는 것이 중요하다.** 

아래 사진은 논문에서 발표한 수식이다. 

![image-20220131223827810](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131223827810.png)

alpha, beta, gamma는 small grid search로 결정할 수 있는 상수이다. 

파이는 사용자가 지정하는 coefficient로 컴퓨터 자원을 얼마나 더 사용할 수 있는지를 나타낸다. 
$$
\alpha*\beta^2 * \gamma^2의 \;의미는\; \\
depth를 \;k배\; 늘렸을 \;때\; FLOPS도 \;K배\; 늘어나지만, \\
width와 \;resolution은 \;k^2배로 \;늘어나는 \;것을\; 의미한다.
$$
마지막 수식에 phi제곱을 취하면 아래 식이 나온다. 
$$
(\alpha*\beta^2 * \gamma^2)^\phi\approx 2^\phi
$$
만약 FLOPS를 2^phi만큼 더 사용할 수 있다면, 위 수식으로 alpha, beta, gamma를 구하게 된다. 

![image-20220209152711426](../AppData/Roaming/Typora/typora-user-images/image-20220209152711426.png)

위 단계를 거쳐서 적절한 값을 찾는다. 

알파, 베타, 감마를 작은 모델에서 찾는 것이 큰 모델에서 찾는 것보다 당연히 좋진 않지만, 큰 모델에서 찾는 경우는 cost가 너무 들기 때문에 작은 모델에서 찾는다. 

# 4. EfficientNet 구조

위 scaling방식을 입증하기 위해서 새로운 mobile-size baseline인 EfficientNet을 만들었다. ([3]을 참고)

아래 사진은 EfficientNet-B0 이다. 

![image-20220209152621083](../AppData/Roaming/Typora/typora-user-images/image-20220209152621083.png)

![image-20220131231805272](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131231805272.png)

> 여기서 SE가 이해가 안됏는데, squeeze and excitation이라고 SENet을 보면 된다. 
>
> senet에서는 global average pooling을 통해서 각 채널의 HxW사이즈를 하나의 값으로 pooling하고, W1을 곱해서 ReLU를 취하고 W2를 곱해서 sigmoid를 취한다. 
>
> 이렇게 나온 값은 0~1의 값을 가지게 되고, 각 채널의 중요도를 의미한다. 
>
> 그리고 원래 입력에다가 이 값들을 곱해주면 (위 그림에서 MUL부분)각 채널의 중요도를 가지고 있는 새로운 input이 되어 다음 layer로 들어간다. 
>
> SEblock이 아래 그림임![image-20220131233008867](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131233008867.png)

모델을 scale up하는 최종 목적은 아래 식을 optimize하는 것이라고 할 수 있다. 

![image-20220131224312952](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131224312952.png)

$$
ACC(m) : 모델의\; 정확도\\
T : 목적하는\; FLOPS\\
w : ACC와\; FLOPS의\; trade\; off를\; control하는\; hyperparameter
$$
**`Flops`** 란 floating point operations per second로 컴퓨터의 성능을 수치로 나타낼  때 주로 사용되는 단위이다. 

초당 부동소수점 연산이라는 의미로 컴퓨터가 1초동안 수행할 수 있는 부동소수점 연산의 횟수를 기준으로 삼는다. 

하지만, 딥러닝에서는 단위 시간이 아닌 절대적인 연산량 (x, + 등)의 횟수를 지칭한다. 

논문에서는 w를 -0.07로 사용하였다. 

> [3]번 논문에서 사용한 값을 그대로 사용
>
> 간단하게 설명하자면, 원래 저 논문에서는 FLOPS로 안하고 Latency로 계산을 진행했다. 
>
> 실험적으로, latency가 2배 증가했을 때, 정확도가 5% 증가했다고 한다. 
> $$
> M1 : latency = l,\; acc= a\\
> M2 :latency = 2l,\; acc=a(1+0.05)\\
> Reward(M1)=a(l/T)^w\approx a(1+0.05)(2l/T)^w=Reward(M2)
> $$
> 위 수식을 풀면 w는 -0.07이 나온다. 
>
> latency보다 FLOPS를 사용하는 이유는 특정 device를 target으로 하는 것이 아니기 때문이다. (latency란 모델이 한 번 돌아가는데 걸리는 지연시간)



# 5. 결과

### 5-1. 다양한 EfficientNet 실험 (phi를 변경하며 실험)

$$
\alpha=1.2,\;\;\beta=1.1\;\;\gamma=1.15로\; 고정하고 \;\phi만\; 변경
$$

![image-20220131224900038](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131224900038.png)



### 5-2. MobileNets과 ResNet에서의 실험

![image-20220131224921321](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131224921321.png)

### 5-3. FLOPS per Accuracy

![image-20220131224929075](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131224929075.png)

### 5-4. For Transfer Learning

![image-20220131224940173](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131224940173.png)

### 5-5. Multi dimension scale up

![image-20220131225006902](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220131225006902.png)





[1] Zoph, B. and Le, Q. V. Neural architecture search with reinforcement learning. ICLR, 2017

[2] He, Y., Lin, J., Liu, Z., Wang, H., Li, L.-J., and Han, S. Amc: Automl for model compression and acceleration on mobile devices. ECCV, 2018.

[3] Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., and Le, Q. V. MnasNet: Platform-aware neural architecture search for mobile. CVPR, 2019.

