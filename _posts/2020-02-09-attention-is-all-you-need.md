---
layout: post
title: "Attention is All You Need - Paper Review"
subtitle: BERT의 기반 네트워크, Transformer
tags:
    - Paper Review
    - Machine Learning
---

BERT의 기반 네트워크인 Transformer를 소개한 논문입니다.
BERT 논문을 본격적으로 분석해보기 전에 Attention과 Transformer 자체에 대한 이해를 더욱 깊게 하기 위해서 매우 자세하게 정리했습니다.

[ArXiv에서 다운로드받을 수 있습니다.](https://arxiv.org/abs/1706.03762)

<hr>

# Introduction

기존에는 Language Modeling이나 Machine Translation과 같은 변환 문제(Transduction Problem)들을 해결하는데 RNN을 사용하곤 했는데, 다음과 같은 문제점들이 있습니다.

-   Sequential한 특성 때문에 병렬화 (Parallelization)를 하지 못함
-   Sequence Length를 가진 훈련 데이터 학습이 어려움
-   많은 데이터를 한 번에 학습시키기에 힘든 Memory Constraints가 존재함

최근에 [LSTM에서의 Factorization Trick](http://arxiv.org/abs/1703.10722)이나 [Conditional Computation](http://arxiv.org/abs/1701.06538)등을 통해서 많은 발전이 있었고, 특히 Conditional Computation은 Model Performance를 크게 끌어올리는데 좋은 효과가 있었습니다.
그래도, **Sequential Computation이 주는 제약인 두 번째 문제점은 여전히 존재했습니다.**

그 이후 **[Attention Mechanism](https://arxiv.org/abs/1702.00887)**이 등장했고, 우리의 눈길을 끌었던 변환 모델이나 Sequence Modeling 모델들의 필수적인 요소가 되었습니다.
제일 좋았던 점은, Input Sequence와 Output Sequence에서 문장 구성 요소들 간의 의존 길이가 얼마나 긴 지에 관계없이 학습이 잘 이루어졌다는 것입니다.
하지만 여전히 대부분의 Attention Mechanism을 차용한 모델들은 RNN과 결합되어 사용되고 있었습니다.

이 논문에서는 Transformer를 소개합니다.
Transformer는 **RNN을 완전히 뜯어내고 오직 Attention만을 사용해서** Input과 Output 사이에 의존성을 형성하도록 한 새로운 Model Architecture입니다.
Transformer는 굉장히 많은 병렬화를 가능하게 하고, Translation Quality 면에서 SOTA(State-of-the-art)를 달성했습니다.
Tesla P100 8개를 사용해서 12시간 훈련했다고 합니다.

<hr>

# Background

Sequential Computation을 줄이려는 목표는 이전부터 있어왔고, 기본적으로 CNN을 사용해서 병렬적으로 계산하려고 했던 시도가 많았습니다.
Extended Neural GPU, ByteNet, 그리고 ConvS2S 등이 이에 해당합니다.
하지만, 이 경우에는 Input과 Output에서 두 임의의 위치 간에 의존성을 만들려고 할 때 **그 위치 사이의 거리가 멀수록 더욱 많은 계산**을 하게 합니다.
각 모델 별로 스케일은 차이가 있지만, 결국 **거리가 먼 위치 간 의존성 학습을 방해하는 요소로 작용**했습니다.
ConvS2S의 경우는 $$O(n)$$, ByteNet은 $$O(n \log n)$$ 스케일로 커지게 됩니다.

Transformer는 이 연산을 상수 횟수의 계산으로 줄여버렸고, 대신 Attention-Weighted Position에 평균을 취해주면서 이 부분에서 취하던 효율성을 포기했습니다.
그러나, 후에 서술할 **Multi-Head Attention**을 통해 효율성 저하를 상쇄시켰습니다.

또, Transformer는 Self-Attention을 사용했습니다.
Self-Attention은 하나의 Sequence에서 각 요소를 **자기 자신의 다른 부분 요소와 의존 관계를 이어주게 하는 Attention Mechanism**입니다.
이는 독해, 추상적 요약, Textual Entailment, 혹은 수행하고자 하는 일에 의존하지 않는 문장 표현 학습에서 큰 효과를 거두었습니다.

> ##### Textual Entailment?
>
> Textual Entailment란, 문장 2개가 주어졌을 때 한 문장으로부터 다른 문장의 의미가 도출될 수 있는가를 보는 Task입니다. 예컨대, "친구는 집에 강아지를 키운다"가 "친구는 집에 동물을 키운다"라는 의미로 도출되는지를 확인합니다. LSTM과 3-way Softmax Classifier를 사용해서 Accuracy 83.5%를 달성한 바 있습니다. ([논문을 참조해보세요.](https://arxiv.org/abs/1509.06664))

<hr>

# Model Architecture

Transformer는 다음과 같은 Encoder-Decoder 모델을 사용합니다.

{% include image.html url="/img/in-post/attention-is-all-you-need/transformer-model.png" description="Transformer Model의 구조" %}

## Encoder and Decoder Stacks

### Encoder

Encoder는 $$N = 6$$개의 똑같이 생긴 Layer로 구축되고, 각 Layer는 각각 2개의 Sublayer를 가지게 됩니다.
첫 번째는 **Multi-Head Self-Attention**이고, 두 번째는 **Position-Wise Fully Connected Feed-Forward Network**입니다.
각 Sublayer 별로 Residual Connection이 이어져있고, Layer Normalization 역시 붙어있습니다.
그래프에도 나와있듯, 만약 각 Sublayer의 내부 구현을 $$\text{Sublayer}(x)$$로 표기한다면, 각 Sublayer는 $$\text{LayerNorm}(x + \text{Sublayer}(x))$$의 결과를 내게 됩니다.
Residual Connection을 이어주려면 더하는 과정에서 각 Tensor의 차원이 같아야 하니까, 모든 Sublayer는 $$d_{model} = 512$$의 동일한 차원의 Output을 갖습니다.

### Decoder

Decoder 역시 $$N = 6$$개의 똑같이 생긴 Layer로 구축됩니다. 총 3개의 Sublayer를 가지고, 순서는 다음과 같습니다.

1. **Masked Multi-Head Attention**
2. **Multi-Head Attention**
3. **Position-Wise Fully Connected Feed Forward Network**

Encoder와 비슷하게 생겼는데, 맨 앞에 Masked Multi-Head Attention이 붙은 꼴입니다.
역시 Residual Connection은 각 Sublayer에 붙어있고, Layer Normalization도 똑같이 붙어있습니다.

한 가지 다른 점은, 여기는 Decoder이기 때문에 추론 과정에서 현재 추론하고 있는 부분 뒤쪽의 Attention Weight에 영향을 받아서는 안된다는 점입니다.
따라서, 각 Time Step에 따라서 의도적으로 문장의 뒷 부분을 가려줘야(Masking) 합니다.
만약 $$i$$번째 결과를 뽑아내고 있다면, 그 상태에서는 $$i$$보다 작은 위치의 결과만을 사용해야 한다는 뜻입니다.

## Attention

Attention은 미리 구축해놓은 Key-Value 쌍이 있을 때 모든 Key 값에 대하여 주어진 Query와의 연관성을 알아내고, 이 값과 Value를 가중합(Weighted Sum)하는 역할을 합니다.
연관 관계는 Softmax가 취해져 나오므로 확률 분포로 여겨지게 됩니다.
여기서 Query, Key, Value, 그리고 Attention 자체의 결과는 모두 Vector입니다.

{% include image.html url="/img/in-post/attention-is-all-you-need/attentions.png" description="Scaled Dot-Product Attention과 Multi-Head Attention" %}

### Scaled Dot-Product Attention

이 논문에서는 Scaled Dot-Product Attention을 정의합니다.
Input으로는 다음 값들이 들어옵니다.

-   $$Q \in \mathbb{R}^{d_k}$$ : Query 값
-   $$K \in \mathbb{R}^{d_k}$$ : Query 값
-   $$V \in \mathbb{R}^{d_v}$$ : Query 값

위 입력이 주어졌을 때, Attention을 수행하는 함수 $$\text{Attention}(Q, K, V)$$는 다음과 같이 정의됩니다.

$$
\text{Attention}(Q, K, V)
=
\text{softmax}(
    \frac
    {QK^T}
    {\sqrt{d_k}}
)V
$$

우선 $$Q$$에 $$K$$를 전치하여 곱함으로써 Query와 Key 간의 유사도를 계산합니다.
그 후 이 값을 $$\sqrt{d_k}$$로 나누어주고 있는데, 왜 그런 걸까요?

가장 많이 사용되는 Attention Function으로는 **Additive Attention**과 **Dot-Product Attention**이 있습니다.
Additive Attention은 한 개의 Hidden Layer를 가진 Feed Forward Network를 통해 Compatibility Function을 계산합니다.
Dot-Product Attention은 Transformer에서 사용하는 알고리즘과 유사하지만, $$\frac1{\sqrt{d_k}}$$를 곱해주는 부분만이 다릅니다.
이론적으로는 두 알고리즘이 비슷한 복잡도를 가지지만, 고도로 최적화된 행렬곱 계산 덕분에 실제적으로는 Dot-Product Attention이 훨씬 빠르고 공간 측면에서 효율적이게 됩니다.

그런데, 조사 결과 $$d_k$$값이 어느 정도의 크기를 갖느냐에 따라 두 알고리즘의 성능이 크게 달라졌던 겁니다.
이 값이 작을 경우 두 알고리즘 모두 비슷한 성능을 냈지만, **$$d_k$$값이 클 경우 Additive Attention 방식이 훨씬 좋은 성능을 보였습니다.**
저자들은 $$d_k$$가 커질 경우 Dot Product 값이 확연히 커지게 되고, Softmax를 취했을 때 **값이 그래프의 양 끝 쪽에 찍히면서 Gradient가 매우 작아지는 현상**이 벌어질 것으로 의심했습니다.
이 문제에 대응하기 위해서 적절한 값으로 결과값을 나누어줌으로써 전체적으로 값의 크기를 줄이는 식으로 대응하게 되었고, 그 적절한 값으로 $$d_k$$가 선택되었습니다.

연산이 끝나고 Softmax를 취하게 되면, 전체 합이 1로 구성된 확률 분포의 속성을 띤 Vector가 나오게 됩니다. 이 값에 $$V$$를 곱해 각 시점에서의 Context Vector를 생성하고 이 값을 반환합니다.

### Multi-Head Attention

Transformer는 Scaled Dot-Product Attention을 사용하는데 그치지 않고 조금 더 나아갔습니다.
단순히 $$d_{model} = 512$$ 크기의 Key, Value, Query들을 사용하는 대신, 조금 더 작은 차원으로 각 값들에 대해 회 Linear Projection을 수행하고 이 값들에 대해 각각 병렬적으로 Scaled Dot-Product Attention을 수행하도록 변경했습니다.
그 결과물들은 전부 모아 이어붙이고(Concat) 그대로 다시 Linear Projection을 수행해서 최종 결과에 이르게 됩니다. 여기에서 각 Linear Projection은 학습되는 Parameter로 존재합니다.
이 방식을 사용하면, 모델이 각각 다른 위치의 각각 다른 표현 Subspace들에서 온 정보들을 골고루 살펴보게 할 수 있었습니다.
이걸 1개만 하게 되면, 평균을 내는 과정에서 장점을 잃게 됩니다.
이 방식을 $$h$$개의 Head를 사용한다고 표현하여 **Multi-Head Attention**이라고 합니다. 식은 다음과 같습니다.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \\

\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

이 식을 구성하는 각 Parameter의 차원은 다음과 같습니다.

$$
W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k} \\
W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \\
W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

위 값들로 Linear Projection이 진행되면서 크기가 $$d_{model} = 512$$였던 $$Q, K, V$$가 각각 Dot-Product Attention의 입력 크기에 맞게 변환되는 것입니다.
기존 Single-Head Attention과 비교할 때, 이 방법은 차원이 줄어든 상태에서 행렬곱을 수행하기 때문에 전체 Computational Cost는 비슷하게 됩니다.

논문에서는 Head의 갯수인 $$h$$ 값을 8로 설정했습니다.
또한 $$d_k = d_v = d_{model}/h = 64$$로 설정하고 학습시켰다고 합니다.
전체 차원을 $$h$$개로 쪼개어 각각 학습시킨 셈이라고 해석할 수 있겠습니다.

### Applications of Attention in our Model

Transformer는 Multi-Head Attention을 다음의 세 가지 방법으로 사용했습니다.

1. Encoder-Decoder Attention Layer

    - Decoder의 모든 위치에서 Input Sequence의 모든 부분에 주의를 기울일 수 있게 조절함

2. Encoder 안의 Self-Attention Layer

    - Encoder의 모든 위치에서 Encoder의 직전 Layer의 모든 부분에 주의를 기울일 수 있게 조절함

3. Decoder 안의 Self-Attention Layer
    - Decoder의 모든 위치에서 지금까지의 모든 Output과 자기 자신의 위치 부분에 주의를 기울일 수 있게 조절함

3번의 경우, Decoder의 Auto Regressive한 속성을 지키기 위해서 Scaled Dot-Product Attention의 불필요한 가중치를 없애주기 위해서 $$- \infty$$로 불필요한 값들을 Mask했습니다.

## Position-wise Feed Forward Networks

Encoder와 Decoder는 각각 모든 위치에 분리되어 똑같이 적용되는 Fully Connected Feed Forward Network를 지닙니다.
수식은 다음과 같습니다.

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

두 번의 선형 변환이 이루어지고, 두 변환 사이에 ReLU Activation이 이루어지고 있습니다. 이 변환이 이루어지는 위치는 모든 Layer가 같지만, 각 Layer마다 Parameter는 별도로 학습합니다.
이 과정은 Kernel Size가 1로 설정된 2번의 Convolution으로도 해석되는데, Input과 Output은 각각 $$d_{model} = 512$$로 설정됩니다.
안쪽의 Layer는 $$d_{ff} = 2048$$의 차원을 갖습니다.

## Embeddings and Softmax

다른 Sequential한 변환 모델들처럼, Transformer도 학습된 임베딩(Learned Embeddings)을 사용해서 Input Token과 Output Token들을 $$d_model$$ 크기로 변환합니다.
또한 일반적인 학습된 선형 변환과 Softmax 함수를 사용해서 Decoder의 실행 결과를 다음 Token의 확률분포로 변환합니다.
여기에는 전부 동일한 가중치 Matrix를 사용하며, 임베딩 Layer 부분에만 각 가중치에 $$\sqrt{d_{model}}$$을 곱한 값을 사용합니다.

## Positional Encoding

Transformer는 RNN이나 Convolution과 같이 순서 정보를 사용하는 Layer가 없기 때문에, 뭔가 문장에서의 위치를 절대적이든 상대적이든 삽입해주어야만 합니다.
저자들은 입력값 임베딩에 Positional Encoding을 도입해서 위치 정보를 삽입하였습니다.
이 값은 $$d_{model}$$차원을 갖도록 해서 임베딩 결과 Vector와 같은 차원을 지니며, 그렇기에 덧셈 연산을 수행할 수 있습니다.
본래 Positional Encoding은 학습시키기도 하고, Fixed로 운용하기도 합니다.

여기에서는 주기가 다른 사인 함수와 코사인 함수를 사용했습니다.

$$
PE_{(pos, 2i)} = \sin(\text{pos}/10000^{2i/d_{\text{model}}}) \\
PE_{(pos, 2i+1)} = \cos(\text{pos}/10000^{2i/d_{\text{model}}})
$$

여기서 $$\text{pos}$$는 위치를 나타내며 $$\text{i}$$는 차원의 Index를 의미합니다.
즉 각 Positional Encoding의 차원은 사인 곡선과 상응하는 관계를 지니게 됩니다.
저자들은 이런 형태의 주기 함수가 요소 간의 상대적인 위치에 주의를 기울이도록 쉽게 학습시킬 수 있다는 가설을 세웠습니다.

이외에도, 저자들은 학습된 Positional Encoding 역시 실험했습니다.
그 결과 두 개의 버전이 비슷한 결과를 냈고, 사인 곡선을 사용하는 쪽이 만약 학습 때 보지 못했던 정말 긴 Sequence를 마주하더라도 추론이 원활하게 이루어질 수 있다고 생각해서 이쪽을 골랐습니다.

<hr>

# Why Self-Attention

저자들은 세 가지의 효과를 원하고 Self-Attention을 사용했습니다.

-   Layer 별 전체 계산 복잡도
-   병렬화할 수 있는 계산의 양 (순서적 계산 횟수의 최솟값으로 계산했을 때)
-   긴 거리 간의 의존성 설계에서의 Path Length

긴 거리 간의 의존성 학습은 많은 Sequence 변환 작업의 주요한 문제입니다.
이 의존성을 학습하는 능력에 영향을 끼치는 큰 요소는, 정방향 및 역방향 신호가 얼마나 긴 거리를 탐색해야 하는지 입니다.
이 값이 짧으면 짧을수록 긴 거리 간 의존성을 학습하기에는 더 쉬워집니다.
그래서 저자들은 이 Maximum Path 값을 각각 다른 Layer 형식들에 대해 측정했습니다.

| Layer의 종류          | Layer별 복잡도   | 순서적 계산 빈도 | Maximum Path    |
| --------------------- | ---------------- | ---------------- | --------------- |
| Self-Attention        | $$O(n^2 * d)$$   | $$O(1)$$         | $$O(1)$$        |
| Recurrent             | $$O(n * d^2)$$   | $$O(n)$$         | $$O(n)$$        |
| Convolutional         | $$O(k * n * d)$$ | $$O(1)$$         | $$O(log_k(n))$$ |
| 제한된 Self-Attention | $$O(r * n * d)$$ | $$O(1)$$         | $$O(n/r)$$      |

위 표에서 보이듯, Self-Attention을 사용하면 Maximum Path Length가 $$O(1)$$이 되어 어느 위치이던 상관없이 상수 횟수만큼의 순서적 계산으로 연결할 수 있습니다.
RNN을 사용했을 때는 이 되어 굉장히 느리게 됩니다.

계산 복잡도 면에서 생각해보았을 때, $$n < d$$가 성립하면 Self-Attention Layer가 Recurrent Layer보다 빠른 속도로 동작합니다.
중요한 건 이 경우가 SOTA를 찍은 대부분의 기계 번역 작업의 문장 표현(Sentence Representation)에 해당된다는 것이죠.
하지만 당연히 이 경우를 벗어나는 경우가 생길 수 있습니다.

굉장히 긴 Sequence를 다룰 때의 계산 성능을 증가시키기 위해서, Input Sequence를 볼 때 내 위치에서 $$r$$칸 만큼만 떨어진 이웃 요소들만 보게 제한할 수 있습니다.
그렇게 되면 Maximum Path Length가 표에서 보이듯 $$O(n/r)$$로 늘어나게 됩니다.
이 부분은 Future Work에서 다룬다고 합니다.

{% include image.html url="/img/in-post/attention-is-all-you-need/multi-head-attention.png" description="Multi-Head Attention의 시각화" %}

위에서 설명한 3개의 이유와 별개로, Self-Attention을 사용하면 더욱 해석 가능한 모델을 만들 수 있습니다.
우리는 이런 형태의 모델을 썼을 때 Attention이 각각 어떻게 연결되었는가를 시각화할 수도 있고, 내부에서 문법적이거나 의미론적인 구조가 잡히는지도 확인해볼 수 있습니다.

<hr>

# Training

## Training Data and Batching

저자들은 WMT 2014 English-German Dataset을 사용해서 모델을 학습시켰습니다.
4,500,000개의 문장 쌍으로 구성되어있는 병렬 코퍼스입니다.
Byte-pair Encoding을 사용하여 인코딩되었고, 37,000개의 Token들로 이루어진 Source-Target Vocabulary가 사용되었습니다.
English-French의 경우에는 훨씬 많은 36,000,000개의 문장 쌍을 총 32,000개의 Word-Piece Vocabulary로 분절한 것을 사용했습니다.
각 문장들은 길이가 비슷한 것들끼리 묶였고, 각각 배치는 얼추 25,000개의 Source Token과 25,000개의 Target Token으로 구성되었습니다.

## Hardware and Schedule

8개의 NVIDIA Tesla P100이 붙어있는 한 개의 머신을 사용해서 학습을 진행했습니다.
Hyperparameter들은 논문에 나온 그대로를 사용했으며, 각 Training Step은 약 0.4초 정도의 시간이 소요되었고, 100,000 step을 돌리는 동안 12시간이 소요되었습니다.
큰 버전의 모델의 경우에는 각 Step당 약 1초, 전체 300,000 step을 약 3.5일간 학습시켰습니다.

## Optimizer

저자들은 Adam Optimizer를 사용했고 $$\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$$값을 부여했습니다.
Learning Rate는 학습이 진행됨에 따라 다음 공식으로 정의됩니다.

$$
\text{lrate} = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5})
$$

첫 $$\text{warmup\_steps}$$동안은 선형적으로 증가하다가, 그 뒤로는 Step number의 Inverse Square Root로 줄어들게 됩니다.
논문에서 제시하는 값은 4,000입니다.

## Regularization

논문은 세 가지 형식의 Regularization을 적용했습니다.

### Residual Dropout

각 Sublayer마다 Dropout을 적용하고 나서 Normalize된 후 다음 Sublayer에 Input으로 들어가게 설정되었습니다. 또한 임베딩의 전체 합과 Positional Encoding에도 Dropout이 동일한 비율로 적용되었습니다. Base Model에 대해서 기본 값은 $$P_{drop} = 0.1$$입니다.

### Label Smoothing

학습이 진행되는 동안 $$\epsilon_{ls} = 0.1$$으로 Label Smoothing이 적용되었습니다. Perplexity 값에는 안 좋은 영향을 끼치지만, Accuracy와 BLEU Score에는 향상이 있었습니다.

<hr>

# Results

{% include image.html url="/img/in-post/attention-is-all-you-need/results.png" description="실험 결과 기존 모델들을 월등하게 앞서고 있습니다." %}

WMT 2014 English-German Translation Task에서, BLEU 점수 28.4를 기록하며 기존 모델들의 BLEU 점수를 2.0 정도 상승시킨 수준의 SOTA를 찍었습니다.
심지어 이 결과는 학습에 3.5일밖에 소모하지 않았음에도 불구하고 Base Model이 모든 모델보다 앞선 결과를 보였습니다.

WMT 2014 English-French Translation Task에서, Big Model이 다른 모델보다 약 4분의 1 수준의 Training Cost를 사용했음에도 불구하고 BLEU 점수 41.8을 기록하며 SOTA를 찍었습니다.

Base Model의 경우 마지막 5개의 Checkpoint를 평균하여 사용하였고, Big의 경우 20개를 평균해서 사용했습니다.
Beam Size 4와 Length Penalty $$\alpha = 0.6$$을 적용한 Beam Search를 사용했습니다.
Maximum Output Length는 Input Length + 50의 수준으로 결정했으나, 가능할 경우 Early Termination 하도록 설정했습니다.

## Model Variations

{% include image.html url="/img/in-post/attention-is-all-you-need/model-variations.png" description="다양한 Hyperparemeter들간의 비교" %}

(A)에서는 Attention Head의 갯수 $$k$$를 변경해보았고, 너무 큰 값을 설정했을 경우에는 되려 BLEU 점수가 떨어지는 결과를 보였습니다.
(B)에서 $$d_k$$값을 변경하는 실험을 한 결과 크기를 줄이면 모델의 품질에 악영향을 끼친다는 결론을 얻었고, Dot Product보다 조금 더 복잡한 Compatibility Function을 사용하는 쪽이 더 이로울 것이라는 가설을 세웠습니다.
(C)와 (D)에서는 모델의 크기와 Dropout을 보았는데, 큰 모델일 수록 더 나은 성능을 보였으며, Dropout은 Overfitting을 막는데 굉장히 효과적이었습니다.
(E)에서는 사인 함수를 활용한 Positional Encoding을 학습되는 형태로 바꾸었는데, Base Model과 거의 다름이 없는 결과를 보였습니다.

## English Constituency Parsing

{% include image.html url="/img/in-post/attention-is-all-you-need/english-constituency.png" description="다른 Task에서도 잘 Generalize되는 것을 볼 수 있습니다." %}

Transformer 모델이 다른 Task에도 효율적으로 쓰이는지를 검증하기 위해 Wall Street Journal 데이터를 이용해서 $$d_{model} = 1024$$ 크기의 4-layer Transformer를 구축했고, Task-specific Tuning을 가하지 않았음에도 불구하고 타 모델들과 비슷한 결과를 낼 수 있었습니다.

RNN Seq2Seq 모델과 대비했을 때, 40,000개의 문장만으로 이루어진 WSJ 데이터를 사용했음에도 불구하고 Berkeley-Parser보다 더욱 좋은 성능을 볼 수 있었습니다.

<hr>

# Conclusion

이 논문에서는 완전히 Attention에만 의존하는 최초의 순서 데이터 변환 모델인 Transformer를 소개했습니다.
RNN을 완전히 걷어내었고, Encoder-Decoder 모델에 Multi-Head Self-Attention만을 사용했습니다. 여러 평가 지표에서 굉장히 좋은 성적을 냈고, 다른 Task에 적용하려고 준비 중입니다.

추후에는 Restricted Attention Mechanism을 활용해서 이미지나 오디오, 비디오 등의 큰 Input과 Output을 효율적으로 핸들링할 수 있는 구조를 실험할 예정입니다.
Tensorflow 기반 구현체를 [GitHub](https://github.com/tensorflow/tensor2tensor)에서 찾을 수 있습니다.
