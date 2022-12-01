import cv2
import numpy as np

bg = cv2.createBackgroundSubtractorMOG2()

def cv_setup(game):
    cv_init(game)
    cv_update(game)

def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)

def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    image = image[:, ::-1, :]
    # cv_process(game, image)
    cv_process_second(game, image)
    # cv_output(image)
    game.after(1, cv_update, game)


def cv_process(game, image):
    h_min = 188 / 2
    h_max = 252 / 2
    s_min = 80
    s_max = 255
    v_min = 2
    v_max = 255

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_blue = np.array([h_min, s_min, v_min])
    high_blue = np.array([h_max, s_max, v_max])
    blue_mask = cv2.inRange(image_hsv, low_blue, high_blue)

    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((2, 2), np.uint8)
    erode = cv2.erode(blue_mask, kernel_ero)
    dilate = cv2.dilate(erode, kernel_dil)

    left = np.sum(dilate[:, :int(dilate.shape[1] / 2)])
    right = np.sum(dilate[:, int(dilate.shape[1] / 2):])

    if left > right:
        game.paddle.move(-10)
    elif right > left:
        game.paddle.move(10)

def cv_process_second(game, image):
    bg_frame = bg.apply(image)
    image_cpy = image.copy()

    ret, thresh = cv2.threshold(bg_frame, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggestArea = 0
    cx = 0
    biggestContour = None

    if len(contours) != 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 15000:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                if area > biggestArea:
                    biggestArea = area
                    biggestContour = contour
                x, y, w, h = cv2.boundingRect(biggestContour)
                cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("windowbg", bg_frame)
    cv_output(image_cpy)

    if cx > 300:
        game.paddle.move(10)
    elif cx < 300:
        game.paddle.move(-10)
    else:
        game.paddle.move(0)

def cv_output(image):
    cv2.imshow("Image", image)
    cv2.waitKey(1)
