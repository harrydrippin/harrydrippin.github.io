---
layout: post
title: "TensorFlow TPU 학습 101"
subtitle: Google ML 서비스의 심장으로 학습하기
date: 2020-04-17
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Engineering
    - Machine Learning
---

이 글은 [Pingpong 블로그](https://blog.pingpong.us/)에 기고된 글로, TensorFlow 2.0으로 TPU 상에서 학습을 시작하기 위해 필요한 과정에 대해 서술한 글입니다. 기록을 위해 제 블로그에도 올려놓습니다.

<hr>

핑퐁팀은 PyTorch뿐만 아니라 TensorFlow 2.0도 활발하게 사용하고 있습니다. 최근 팀 내에서 TPU를 활용하여 학습하는 프로젝트를 진행하고 있는데, 이 글에서는 TF 2.0을 기반으로 짜여진 모델을 Google Colab과 Google Cloud Platform (GCP)을 활용하여 어떻게 TPU 위에서 훈련하는지 정리했습니다.

## 목차

-   [TPU, Google ML 서비스의 심장](#tpu-google-ml-서비스의-심장)
-   [어떻게 사용해야 할까?](#어떻게-사용해야-할까)
    -   [XLA (Accelerated Linear Algebra)](#xla-accelerated-linear-algebra)
    -   [TensorFlow에서 TPU 사용하기](#tensorflow에서-tpu-사용하기)
        -   [TPUEstimator](#tpuestimator)
        -   [TPUStrategy](#tpustrategy)
-   [모델을 학습시켜보자!](#모델을-학습시켜보자)
    -   [Google Colab에서 TPU 사용하기](#google-colab에서-tpu-사용하기)
    -   [GCP에서 TPU 사용하기](#gcp에서-tpu-사용하기)
-   [TensorFlow Research Cloud (TFRC)](#tensorflow-research-cloud-tfrc)
-   [후기](#후기)
-   [참고](#참고)

## TPU, Google ML 서비스의 심장

{% include image.html url="/img/in-post/tpu-with-tf2-and-gcp/tpu-rack.png" alt="Google 데이터 센터에 꽂혀있는 아름다운 TPU들" description="Google 데이터 센터에 꽂혀있는 아름다운 TPU들"%}

TPU (Tensor Processing Unit)는 Google에서 2016년 중순에 발표한 기계학습에 특화된 하드웨어입니다.
TPU는 ML 모델을 학습시킬 때 발생하는 엄청난 양의 선형대수 연산을 8비트 소수점 연산을 통해 가속화해서 CPU나 GPU에서 학습시킬 때와는 비교도 안되는 수준의 학습 속도를 보입니다.
실제로 ResNet-50 같은 모델을 불러와서 테스트하면, Tesla V100 8개를 사용해서 훈련시켰을 때보다 GCP의 TPU v2 Pod를 사용했을 때 **27배 가량 빠른 속도로 훈련**되었다고 합니다.

TPU의 설계 의도가 애초에 Google에서 이미지 검색과 같은 ML 모델 기반 서비스를 어떻게 하면 빠르게 제공할 수 있을까에서 비롯된 것이다보니, 내부적으로 머신러닝 학습과 추론을 빠르게 할 수 있는 여러 기술이 적용되어있습니다.
예를 들어, TPU의 학습은 모든 매개변수를 On-Chip 고대역폭 메모리에 저장해놓고 In-feed Queue에 있는 배치를 바로 읽는 것으로 진행되기 때문에 GPU에서 사용되는 PCIe 버스의 속도보다 훨씬 빠른 속도로 데이터를 TPU 위로 옮겨서 학습 및 추론을 수행할 수 있습니다.

다만 TPU를 개발한 목적이 Google이 이미지 검색 등의 기계학습 추론 서비스를 빠르게 제공한 것이다보니, 기술 유출을 염려해 특허조차 안 내고 있을 만큼(...) 숨겨져 있으며 오직 Google의 인프라에서만 사용할 수 있습니다.
즉, Google Colab에서 무료로 잠시 대여해서 사용하거나 GCP에서 시간당 비용을 내고 대여하는 형태로 사용해야 합니다.

## 어떻게 사용해야 할까?

> _"그럼 엄청 빠른 GPU 사용하는 것처럼 쓰면 되겠네!"_

위와 같이 생각하셨다면, **큰일날 소리입니다.** TPU를 사용해서 학습하다보면 GPU에서 볼 수 없었던 형형색색의 오류들이 자유분방하게 날뛰는 광경을 맞닥뜨릴 수 있습니다.
여러분의 정신 건강을 위해, TPU를 사용하기 이전에 어떤 원리로 학습이 이루어지는지 미리 소개해드리고자 합니다.

### XLA (Accelerated Linear Algebra)

TPU에서 학습하기 이전에 우리는 그래프를 TPU 위로 올려야 하고, 그 전에 그래프를 **컴파일**해서 적당히 최적화해야 합니다.
TensorFlow나 PyTorch와 같은 머신러닝 라이브러리는 연산 그래프의 각 노드만 최대한 빠르게 실행할 수 있도록 구현되어 있기 때문에, 굳이 할 필요 없는 연산이 중간에 끼어있거나 묶어서 한꺼번에 처리하면 더 빨라지는 연산이 생기기도 합니다.
이를 최적화하지 않는다면 제일 효율적으로 학습한다고 말할 수 없겠죠?

조금 더 자세히 예를 들어보겠습니다.
Softmax를 아래에 간단하게 정의해보겠습니다.

```python
def softmax(logits, dim):
    return exp(logits) / reduce_sum(exp(logits), dim)
```

이 연산을 그냥 실행하면, `exp`나 `reduce_sum`과 같은 연산을 전체 행렬에 각각 수행하게 됩니다.
그러면 `exp(logits)` 구하고, `reduce_sum` 구하고, 앞에 `exp(logits)` 한 번 더 했던 거 불러와서 나누고...
쓸데없는 연산이나 메모리 할당이 많아질 수밖에 없죠.

Google에서는 이런 요구를 만족하기 위해 **XLA (Accelerated Linear Algebra)**라는 라이브러리를 만들었습니다.
XLA는 JIT (Just-in-Time) 컴파일을 통해 실제 런타임에 그래프를 분석해서 최적화할 수 있도록 하며, 합쳐서 계산할 수 있는 건 다 합쳐서 네이티브 기계어 형태로 출력해줍니다.
즉, XLA는 예로 들었던 중복 연산이나 중간 단계에서 메모리를 차지하는 변수 할당을 줄여서 빠른 속도와 적은 메모리 사용으로 그래프를 학습할 수 있도록 돕습니다.
정말 빠르겠죠?

이 컴파일러는 TensorFlow를 위해 만들어져서 TensorFlow만 사용할 수 있었는데, 지금은 [PyTorch에서의 XLA 컴파일을 할 수 있게 해주는 라이브러리](https://github.com/pytorch/xla)가 있기 때문에 PyTorch에서도 TPU를 사용할 수 있습니다. 하지만 TensorFlow에서 사용했을 때에 비해서 상대적으로 느립니다. 이 글에서는 TensorFlow 2.0만 우선 다루어보겠습니다.

XLA에 대한 더 자세한 활용 예시는 [Google Developers Blog의 글](https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html)과 [TensorFlow Dev Summit 2017의 XLA 발표](https://youtu.be/kAOanJczHA0)에서 찾아보실 수 있습니다.

### TensorFlow에서 TPU 사용하기

TensorFlow에서 TPU를 사용하는 방법은 크게 두 가지로 나뉩니다.
하나는 **TPUEstimator**를 사용하는 방법이고, 나머지 하나는 **TPUStrategy**를 사용하는 방법입니다.

#### TPUEstimator

[TPUEstimator](https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/tpu/TPUEstimator)는 TF의 표준 API 중 하나인 [`tf.estimator.Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator?hl=ko)의 확장으로, TF 1.0에서 TPU를 사용하는 방식으로 권장되었습니다.
Estimator는 모델을 반환하는 함수를 래핑해서 Training과 Evaluation, Prediction에 사용되는 여러 가지 명령을 함께 제공할 수 있게 하는 객체입니다.
TensorFlow 내부에 있는 확장 구현체로는 `LinearEstimator`, `DNNEstimator` 등이 있습니다.

TPUEstimator 역시 이와 같은 형식으로, TPU에 사용되는 여러 가지 명령이 함께 제공되는 형태입니다.
모델 함수를 만들고 단순히 넘겨주기만 하면 TPU 코어에 모델을 배포해줍니다.

#### TPUStrategy

[TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy)는 TF 2.0 API에 있는 [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)의 확장입니다.
학습 중 상태 값들과 계산 워크로드의 분산을 위해 설정하는 정책으로, 이 객체를 활용해서 데이터나 연산 부담을 여러 기기에 분산하여 나누어줄 수 있습니다.
대표적으로 값을 모든 기기에 복사하는 `MirroredStrategy`와 하나의 기기에서만 학습하는 `OneDeviceStrategy` 등이 있습니다.
실제 학습 시에는 이 객체들을 자유롭게 교체해서 작업에 알맞은 전략을 선택할 수 있습니다.

TPUStrategy는 TPU에 학습할 때 사용하는 Strategy로, 실제 동작은 `MirroredStrategy`와 같이 동작합니다.
GCP에서 제공하는 Cloud TPU나 TPU Pod에 연결할 때 이 객체를 활용하여 TPU와의 Connection을 잡아줍니다.
TF 2.0에서는 이 방식을 사용하라고 권고하고 있으며, 이 글 역시 이 방법을 사용할 예정입니다.

## 모델을 학습시켜보자!

이제 실제 모델을 학습시켜보겠습니다. 모델을 구현하는 것은 이 글의 범주를 벗어나므로, Keras API를 사용하여 Subclassing된 모델 클래스 `Model`이 있다고 가정해보겠습니다.

TPU에 액세스하는 방법은 크게 2가지로, Google Colaboratory를 사용하거나 GCP에서 Compute Engine Instance를 여는 방법이 있습니다.

### Google Colab에서 TPU 사용하기

Google Colab은 GPU를 무료로 제공해주는 Jupyter Notebook이라고 해서 널리 알려졌는데, 사실 TPU 또한 무료로 제공하고 있습니다.
[Google Colab](https://colab.research.google.com/)에 접속해서 새 노트북을 만든 후, "런타임 > 런타임 유형 변경"에서 하드웨어 가속기를 `TPU`로 변경해야 합니다.

Colab의 가속기가 TPU로 변경되면 Colab 서버가 떠있는 인스턴스에서 TPU가 접근이 가능해집니다.
TPU에는 gRPC 형식으로 접근하게 되므로 특정 IP와 Port를 통해 접근합니다.
셀에 다음과 같이 적고 실행해서 할당된 IP를 확인할 수 있습니다.

```bash
!echo $COLAB_TPU_ADDR
```

확인되었다면, 이 주소를 이용해서 TPU에 접근할 수 있습니다.

```python
# TPU gRPC 접근 URI
TPU_PATH = f"grpc://{os.environ['COLAB_TPU_ADDR']}"

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_PATH)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
```

`TPUClusterResolver`를 통해서 하나 혹은 여러 대의 TPU 집합을 잡습니다.
그 후 `experimental_connect_to_cluster` 함수와 `initialize_tpu_system` 함수를 통해 TPU와 런타임을 연결합니다.

그 후에는 Strategy를 만들어줍니다. 위에 적었던 TPUStrategy에 Resolver를 연결해서 사용 대상 TPU 집합을 데이터 분산 대상에 넣어줍니다.

```python
strategy = tf.distribute.experimental.TPUStrategy(resolver)
```

이제 모델을 학습시키기 위한 준비를 마쳤습니다. `strategy.scope()` Context 아래에서 Metric을 적절히 선언하고 실행해주면 됩니다.

```python
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
model = Model(config)
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

with strategy.scope():
    metric = tf.keras.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.fit(train_dataset, epochs=config.epoch)
```

여기에서 주의해야 할 점이 몇 가지 있습니다.

첫 번째는 Metric을 초기화하는 시점과 Model을 Compile 및 Fit하는 시점은 **Strategy Context 하위에 있어야 한다는 점입니다.** 그렇지 않으면 정상적으로 실행되지 않습니다.

두 번째는, 이 단계에서 **원인 모를 세션 다운이 많이 일어납니다.** Google Colab에 제공되는 TPU는 아무래도 무료이다보니 선점형으로 제공되고 연결이 불안정한 측면이 있습니다.
이 때문에 TPU를 사용하여 중요한 모델을 학습하는 경우에 **Google Colab은 디버깅 환경으로 적절하지 않습니다.**
자세한 오류 내용 없이 TPU가 깨지거나 세션이 혼자 닫히기 때문에 자칫 디버깅 중 코드의 오류로 오인할 가능성이 있습니다.

### GCP에서 TPU 사용하기

안정적인 환경에서 TPU를 사용하려면 GCP에서 직접 비용을 지불하고 빌리는 방법이 있습니다.
TPU는 2020년 4월 17일 기준 가장 저렴한 아이오와(us-central1) 지역에서 v2-8 유형이 시간당 \$4.50 정도를 과금합니다.

결제 계정이 연동된 GCP 프로젝트에서 Cloud Shell을 시동하고 다음과 같이 명령합니다.
이 명령이 실행되는 시점부터 과금이 시작됩니다.

```bash
$ ctpu up --zone=us-central1 --name test-tpu
```

명령이 실행되면 `test-tpu`로 이름지어진 TPU가 붙어있는 n1-standard-1 유형의 Compute Engine이 시동됩니다.
생성된 인스턴스에서 TPU에 접근하려면 이름을 사용해야 하며, 이 값은 기본값으로 인스턴스의 환경 변수에 들어있습니다.
다음 명령을 통해 TPU의 이름을 확인할 수 있습니다.

```bash
$ echo $TPU_NAME
```

그 이후에는 Colab에서 했을 때와 동일한 과정을 통해 진행합니다.

```python
# GCP에서는 TPU 접근에 TPU의 이름을 사용합니다.
TPU_PATH = os.environ["TPU_NAME"]

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_PATH)
# ...
```

모두 완료된 뒤에 추가 과금을 막기 위해 Cloud Shell에 다음과 같이 명령합니다.
이 명령은 `ctpu` 명령으로 생긴 Compute Engine 인스턴스와 Cloud TPU를 동시에 종료합니다.

```bash
$ ctpu delete --zone us-central1
```

## TensorFlow Research Cloud (TFRC)

TPU의 가격은 저렴한 편이 아니며, 개인이 지속적으로 사용하기에는 부담스러운 금액입니다.
Google은 TPU 사용이 부담스러운 연구자를 위해서 [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc)라는 연구자 지원 프로그램을 만들었습니다.
2020년 4월 현재, [공식 사이트 링크](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform?hl=ko)를 통해 신청하면 TFRC가 지원한 연구를 전 세계와 공유한다는 조건 하에 30일 간 다음 리소스들을 **무료로** 제공합니다.

-   On-Demand Cloud TPU v3 5개
-   On-Demand Cloud TPU v2 5개
-   선점형 Cloud TPU v2 100개

신청된 경우 접수한 메일 주소로 승인 이메일을 받을 수 있습니다.
승인 이메일 안에는 프로젝트 별로 지원을 신청할 수 있는 Google Form 주소가 있으며, 여기에 연구하고자 하는 프로젝트를 적어서 보내면 위 리소스를 무료로 사용할 수 있습니다.

## 후기

저는 보통 GPU 상에서 학습을 했었고 TPU는 처음 다루어보았어서 굉장히 신기하고 재미있었습니다. 핑퐁팀은 다양한 모델을 만들며 끊임없이 연구해 나갈 예정이고, 그 과정에서 TPU 사용 또한 적극적으로 검토하고 있습니다. 이후에 TPU를 활용하여 큰 모델을 학습시켜보고, 그 과정을 공유하는 글로 다시 돌아오겠습니다. 긴 글 읽어주셔서 감사합니다!

## 참고

-   [Google Cloud - TPU](https://cloud.google.com/tpu/docs/tpus)
-   [XLA - TensorFlow, compiled - Google Developers Blog](https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html)
-   [PyTorch/XLA - Github](https://github.com/pytorch/xla)
-   [TensorFlow API Docs](https://www.tensorflow.org/api_docs/python)
-   [Google Colab](https://colab.research.google.com/)
-   [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc)
