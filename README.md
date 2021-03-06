# 혐오 표현 탐지 모델 개발

> 혐오 표현을 음절 단위로 탐지하는 모델을 개발하였다. <br>
> 혐오 표현이란 어떤 개인 혹은 집단에 대해 그들이 사회적 소수자의 속성을 가졌다는 이유로 차별, 혐오하거나 차별, 적의, 포격을 선동하는 표현을 뜻한다. (출처: 국가권익위원회) <br>
> 욕설 뿐만 아니라 혐오 표현을 탐지하여 더 혐오 표현인 글을 우선 제거 대상으로 선정해 사이버불링 피해가 없는 안전한 온라인 공간을 만들기 위한 노력들의 보탬이 되고자 한다. 

## 서비스 시연
<img width="80%" src="https://user-images.githubusercontent.com/49083528/173006260-078f2f2c-69c1-4688-835a-64c6f9389b87.gif"/>

## 분석 과정 
1. 혐오 표현 정의
2. 데이터 수집 및 불균형 해결
3. 네이버 클린봇 모델링 실험을 참고하여 5개의 모델 비교
4. 모델 성능 개선을 위한 전처리 재 진행
5. 모델 성능 개선을 위한 음절 단위 토큰화 재 진행 (카카오 브레인의 '음절 토큰화' 활용)
6. 모델 성능 비교
7. Django 활용 시연 화면 개발 

혐오 표현의 경우 "지111랄"과 같이 음절 단위로 표현된 경우가 많아, 카카오의 음절 토큰화로 재 진행해 CNN 모델 적용하였다. 모델 예측 결과, 기존 토큰화 방식 Okt의 morchs과 비교했을 때 음절 토큰화 방식이 acc는 0.01 떨어졌으나, Precision, AUC, F1 score 에서 모두 좋은 성능을 보였다. 

- kakaobrain/kortok: <https://github.com/kakaobrain/kortok>

## 데이터 셋 
스마일 게이트 한국어 혐오 발언 데이터 셋 활용

> 한국어 혐오표현 'UnSmile' 데이터셋은 Smilegate AI에서 22.03월 공개한 악플 및 혐오 발언 데이터셋을 활용하였다. 2019년 1월 ~ 2021년 7월 포털사이트, 커뮤니티 등 웹사이트 게시글을 대상으로 한 데이터셋으로 혐오표현 전문가가 최종 검수하였다. 

- smilegate-ai/korean_unsmile_dataset: <https://github.com/smilegate-ai/korean_unsmile_dataset>
- 혐오표현 : 총 12,068건, 클린: 4,674건 
<br>

데이터 불균형 해결을 위한 추가적인 단발성 대화 데이터셋 생성 

> 부족한 클린 문장을 추가하기 위해 ai hub에서 제공하는 한국어 감정 정보가 포함된 단발성 대화 데이터셋을 사용했다. 총 38, 594건 데이터 중 '혐오' 감정 외에 10,200건을 랜덤 추출하였다. 그 중에서도 팀원들과 2000개씩 나눠 단발성 문장을 매뉴얼하게 확인해 혐오 표현이 포함된 문장은 추가 제거하였다. 결과적으로 총 9,394건의 클린 문장을 추가 사용하였다. 

- AI Hub 한국어 감정 정보가 포함된 단발성 대화 데이터셋: <https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-009>
- 클린 문장 : 총 9,394건
<br>

전체 데이터 셋 <br>

- train set <br>
  혐오 표현 문장 : 11,266건 <br>
  클린한 문장 : 11,266건 
 
- validation set <br>
  혐오 표현 문장 : 2,802건 <br>
  클린한 문장 : 2,802건 
  
- 총 데이터 건수 <br>
  28,136건 

## 분석 결과물 
- [Django code](https://drive.google.com/file/d/1VgwxHWHEhLgGjPrzDrTCpi3n-qkRnY3G/view?usp=sharing)
- [Presentation](https://drive.google.com/file/d/1QGBihiMJ7Yf2dJ3zlM-dCVZRbBHYt_89/view?usp=sharing)
- [Modeling code](https://github.com/sihyeon3523/hatred-stop/tree/main/modeling)
