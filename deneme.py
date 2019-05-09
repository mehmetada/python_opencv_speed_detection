import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import time

DEF_ROI_1_VALUE =  65#np.array([65], dtype="uint8")


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def detect_change_on_roi(frame, roi):
    while True:
        if roi <= DEF_ROI_1_VALUE*0.9:
            return 1
        else:
            return 0


def process_image(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny = cv2.Canny(blur, 10, 150)
    roi_1 = blur[470, 420]
    return canny, roi_1


def check(b):

    if b[len(b)-1] == b[len(b)-2]:
        print(" no Car")
        return 0
    else:
        print("yes car")
        return 1


def main(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video file.")

    count = 0
    a = -1
    b = []
    while (cap.isOpened()):

        ret, frame = cap.read()
        frame = rescale_frame(frame, percent=75)

        # ret equals to false when it reaches the end of a video file
        # use the below if statement to avoid cv2 errors
        if ret == False:
            break

        cv2.line(img=frame, pt1=(460, 490), pt2=(460, 350), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(1030, 490), pt2=(1030, 350), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        #print("0")
        processed_image, roi = process_image(frame)
        if detect_change_on_roi(processed_image, roi):
            #timer başlat
            count += 1


        b.append(count)

        check(b)
        print(count)

            #print(b)
            #print(b[len(b)-1])

        #print(count)
        #print("1")


        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main("FroggerHighway.mp4")
