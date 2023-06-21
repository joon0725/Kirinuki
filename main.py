import cv2
import numpy as np
import tensorflow as tf

# deeplab.ipynb에서 만든 모델을 불러온다
model = tf.keras.models.load_model("model.h5")


# 이미지를 모델 입력 크기에 맞게 변환해주는 함수
def read_image(img):
    img = cv2.resize(img, (512, 512))
    return img

# Semantic Segmentation을 통해 사람이 있는 곳을 확인한 후에 그 영역을 하얀색으로 마스킹해주는 함수
def colormap(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for l in range(1, 20):
        idx = mask == l
        r[idx] = 255
        g[idx] = 255
        b[idx] = 255
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 모델을 적용해서 입력받은 이미지에서 사람이 있는 영역을 Segmentation하는 함수
def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

# 누끼를 딴 최종적인 이미지를 보여주는 함수
def show_img(img, model):
    img = read_image(img)
    prediction_mask = infer(model=model, image_tensor=img)
    mask = colormap(prediction_mask)
    # 입력받은 사진과 마스킹 맵을 bitwise and 연산하여 사람이 있는 부분만 남긴다.
    masked = cv2.bitwise_and(img, mask)
    img = cv2.resize(masked, (640, 480))
    cv2.imshow("videos", img)


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# esc 키가 눌리기 전까지 계속 카메라에서 이미지를 받아와서 누끼를 따는 show_img 함수를 거쳐 출력한다.
while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    show_img(frame, model)

capture.release()
cv2.destroyAllWindows()
