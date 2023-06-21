# Kirinuki
실시간 누끼 따주는 프로그램

# 필요한 모듈 설치
pip install -r requirement.txt를 하여 필요한 모듈을 다운받거나 아래 목록에 있는 모듈들을 다운받는다.
* numpy
* matplotlib
* opencv-python
* scipy
* glob
* tensorflow

# 모델 생성
deeplab.ipynb를 실행하여 model.h5 파일을 생성한다.
이 프로젝트에서 사용된 모델은 DeeplabV3+로 *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* 이라는 논문에서 소개된 모델이다

# 실시간 누끼따기 적용
main.py를 실행하면 실시간 누끼따기를 할 수 있다
