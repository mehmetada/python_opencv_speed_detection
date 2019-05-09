import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import time

DEF_ROI_1_VALUE = [65, 60, 56]
#def draw_lines(video):
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def main(path):
    cap = cv2.VideoCapture(path)

    if (cap.isOpened() == False):
        print("Error opening video file.")

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = rescale_frame(frame, percent=75)

        # ret equals to false when it reaches the end of a video file
        # use the below if statement to avoid cv2 errors
        if ret == False:
            break

        #plt.imshow(frame)
        #plt.show(1)
        # show the frame, press 'q' to quit1
        cv2.line(img=frame, pt1=(460, 490), pt2=(460, 350), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(1030, 490), pt2=(1030, 350), color=(255, 0, 0), thickness=5, lineType=8, shift=0)

        #cv2.rectangle(img=frame, pt1=(460, 350), pt2=(1030, 490), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.imshow("frame", frame)
        roi_1 = frame[470, 350]
        roi_2 = frame[1040:1070, 350:490]
        #if roi_1
        #print(roi_1)
        print(roi_1[0], roi_1[1], roi_1[2])
        print(DEF_ROI_1_VALUE)
        #if np.prod(roi_1) <= np.prod(DEF_ROI_1_VALUE)*0.5:
        #    print("car detected")

        """if (-(DEF_ROI_1_VALUE * np.array([0.25, 0.25, 0.25]))<= roi_1):
            print("car detected")

        elif ((DEF_ROI_1_VALUE * [0.25, 0.25, 0.25]) >= roi_1):
            print("Car detected")"""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main("FroggerHighway.mp4")
