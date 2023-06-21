import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")


def read_image(img):
    img = cv2.resize(img, (512, 512))
    return img


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


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def plot_predictions(img, model):
    img = read_image(img)
    prediction_mask = infer(model=model, image_tensor=img)
    mask = colormap(prediction_mask)
    masked = cv2.bitwise_and(img, mask)
    img = cv2.resize(masked, (640, 480))
    cv2.imshow("videos", img)


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    plot_predictions(frame, model)

capture.release()
cv2.destroyAllWindows()
