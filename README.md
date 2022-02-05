# AI Perfumer:sparkles:

#### ※ 1차 활동 : https://github.com/park-jiha/Makers_Day

## ▶ 창의 과제
### ● 국문 : 컴퓨터 비전 기술을 이용한 얼굴 이미지 분석 및 향수 추천 시스템
### ● 영문 : Facial impression analysis and perfume recommendation system based on computer vision technology

## ▶ 목표
#### 1. Pytorch를 이용한 Face Image Classification model 구축
#### 2. Touch Free 시스템 구현을 위한 Mediapipe 기반 Hand Pose Estimation 기술 구현
#### 3. 도출된 결론을 바탕으로 향수 추천 결과를 Flask를 이용하여 최종 web 서비스로 제공

## ▶ 상세 내용
### ➀ Class 선정
: 가장 대표되는 이미지 6가지를 선정했습니다. ‘Active’, ‘Clean’, ‘Cute’, ‘Elegant’, ‘Natural’, ‘Sexy’ 이며, 남자와 여자가 주는 기본적인 인상도 다르기 때문에 성별도 나누어 class를 분류했습니다. 최종적으로 총 12개의 Class를 선정할 수 있었습니다.

### ➁ 데이터셋 구축
: Face Image Classification을 위한 데이터셋을 구축했습니다. 한국인 얼굴 이목구비, 얼굴형과 관련한 논문을 참고하여 객관적인 class 선정에 중점을 두었습니다. 일반인들보다 개성이 뚜렷한 연예인 사진을 위주로 크롤링을 진행하였으며, 노이즈를 줄이기 위한 얼굴 위주 cropping도 진행했습니다. 또한 학습을 저해할 만한 low quality(측면을 보는 사진, 화질이 매우 낮은 사진, 일정 각도 이상 회전이 많이 된 사진 등) 이미지 데이터는 제외시켰으며, 총 5000여장의 학습 데이터셋을 구축할 수 있었습니다.

### ➂ Face Image Classification 모델
: Pytorch 기반 사전학습 모델 ‘ResNet50’을 이용해 이미지 분류를 시도했습니다. Resnet50 은 ImageNet으로 사전 학습된 모델로, 총 50개의 합성곱 레이어를 갖는 CNN 모델입니다. 앞서 구축한 학습 데이터셋을 전이학습하여 남녀 성별 및 인상 class 분류를 진행했습니다. 여러 사전학습 모델을 실험해보았으며, ResNet50 모델이 높은 성능을 보이고 빠른 성능을 내어 가장 적합할 것이라고 판단했습니다.

### ➃ Hand Pose Estimation 구현
: 구글에서 제공하는 MediaPipe 기술 중 Hand landmark 추출 방법을 활용하였고, 총 21개의 손 마디별 키포인트를 추출할 수 있었습니다. 사용자가 touch free 형태로 질문에 답을 할 수 있도록, 검지 끝 마디의 key point 좌표만을 추출해 이를 실시간으로 tracking하고, 정답 박스 영역에 key point가 들어가면 해당 번호로 선택될 수 있도록 코드를 구현했습니다. 총 세번의 질문에 대한 답은 list에 저장되고 결론 도출에 사용됩니다.

### ➄ 답변 조합에 따른 결과 도출
: 얼굴 이미지 분석에 대한 결과 하나와, 사용자가 직접 질문에 답한 결과로 총 2가지의 결과를 도출합니다. 결과에 적합하는 향수들에 대한 정보를 수집하고 분류했습니다.

### ➅ Flask를 이용한 웹 서비스 제작
: 앞선 모델 및 알고리즘으로 도출된 결과를 웹 서비스로 나타내줍니다. Flask를 이용하여 실시간 웹캠 송출을 함과 동시에 얼굴 분류 모델 및 Hand Pose Estimation이 진행되고, 결과가 도출되면 향수 추천 결과창을 웹 화면에 보여줍니다. 웹 디자인은 CSS를 활용해 진행했습니다.

## ▶ 주간 활동 결과


## ▶ 기대효과
### ➀ Deep Learning Task
: 이미지 분류 모델을 활용해보면서, 사람 얼굴의 작고 미세한 특징 조합을 어느 정도까지 구분해 낼 수 있을지 확인해보는 좋은 시도가 되었습니다. 이를 통해 기존의 분류 문제보다 더 복잡한 문제들을 해결해나가는 발전적인 과정이 될 것이라 생각합니다.

### ➁ Touch Free System
: 터치 프리 시스템의 활용으로 컴퓨터 비전 기술 중 하나인 Hand Pose Estimation 기술을 이용하여 직접적으로 터치하지 않아도 되는, 비대면 체험 부스나 전시회 및 다양한 서비스에 활용될 수 있는 가능성을 갖습니다.

### ➂ Entertainment AI
: 나의 이미지를 인공지능이 분석해주고, 이에 맞는 향수를 추천해준다는 것은 평소에 쉽게 접하지 못하는 서비스입니다. 사용자들이 이 서비스를 통해 만족감을 느끼고, AI에 대한 흥미 있는 접근이 이루어진다면, 그 자체만으로도 긍정적인 재미 요소를 기대해 볼 수 있을 것입니다.

## ▶ 최종 결과물 및 성과 예시
- 사용자가 손으로 질문 결과를 가리키는 장면
<img src="https://user-images.githubusercontent.com/62232217/152659325-c3d7200b-1547-48fa-b382-3651ec633a72.png"  width="600" height="400"/>

- 최종 추천 결과창
<img src="https://user-images.githubusercontent.com/62232217/152659327-fcdae369-a52c-45e9-a37e-53b921388863.png"  width="850" height="500"/>

