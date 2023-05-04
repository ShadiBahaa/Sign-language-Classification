import math

import cv2 as cv
import numpy as np
from Constants import *
from cvzone.HandTrackingModule import HandDetector

capture = cv.VideoCapture(LAPTOP_CAMERA)
hands_detector = HandDetector(maxHands=HANDS_MAX)
folders_counter = ZERO
img_counter = ZERO
while True:
    if folders_counter == CLASSES_MAX:
        break
    if img_counter == SAMPLES_MAX:
        key = cv.waitKey(KEY_DELAY)
        if key == ord('c'):
            continue
        folders_counter += INCREMENT
        img_counter = ZERO
        continue
    done, img = capture.read()
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
                    width_gap = math.ceil((IMG_SIZE - new_width) / 2)
                    white_img[:, width_gap:new_width + width_gap] = img_resized
                else:
                    factor = IMG_SIZE / width
                    new_height = math.ceil(factor * height)
                    img_resized = cv.resize(img_cropped, (IMG_SIZE, new_height))
                    height_gap = math.ceil((IMG_SIZE - new_height) / 2)
                    white_img[height_gap:new_height + height_gap, :] = img_resized
            except Exception as error:
                continue
            cv.imshow("Cropped", img_cropped)
            cv.imshow("White", white_img)
        cv.imshow("original", img)
        key = cv.waitKey(KEY_DELAY)
        if key == ord('s'):
            img_counter += 1
            cv.imwrite("Data/"+folders[folders_counter]+"/"+folders[folders_counter]+"_"+str(img_counter)+".jpg", white_img)
            print("Folder: "+folders[folders_counter]+", Image: ", str(img_counter))
        elif key == ord('e'):
            break
