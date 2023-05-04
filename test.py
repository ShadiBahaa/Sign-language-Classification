import math

import cv2 as cv
import numpy as np
from Constants import *
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


capture = cv.VideoCapture(LAPTOP_CAMERA)
hands_detector = HandDetector(maxHands=HANDS_MAX)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
folders_counter = ZERO
img_counter = ZERO
while True:
    done, img = capture.read()
    img_tmp = img.copy()
    if done:
        hands, img = hands_detector.findHands(img)
        if hands:
            hand = hands[ZERO]
            top_left_x, top_left_y, width, height = hand['bbox']
            white_img = np.ones((IMG_SIZE, IMG_SIZE, COLOR_CHANNELS), np.uint8) * PIXEL_MAX
            img_cropped = img[top_left_y - OFFSET:top_left_y + height + OFFSET,
                          top_left_x - OFFSET:top_left_x + width + OFFSET]
            img_ratio = height / width
            try:
                if img_ratio > IDENTICAL:
                    factor = IMG_SIZE / height
                    new_width = math.ceil(factor * width)
                    img_resized = cv.resize(img_cropped, (new_width, IMG_SIZE))
                    width_gap = math.ceil((IMG_SIZE - new_width) / TWO)
                    white_img[:, width_gap:new_width + width_gap] = img_resized
                else:
                    factor = IMG_SIZE / width
                    new_height = math.ceil(factor * height)
                    img_resized = cv.resize(img_cropped, (IMG_SIZE, new_height))
                    height_gap = math.ceil((IMG_SIZE - new_height) / TWO)
                    white_img[height_gap:new_height + height_gap, :] = img_resized
            except Exception as error:
                continue
            prediction, index = classifier.getPrediction(white_img, draw=False)
            print(prediction)
            cv.putText(img_tmp, folders[index], (top_left_x, top_left_y-OFFSET), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
    cv.imshow("Current Image", img_tmp)
    cv.waitKey(KEY_DELAY)

