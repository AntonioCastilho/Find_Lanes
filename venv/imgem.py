# File name: imagem.py
# Description: Detection of lane demarcation contours.
#              Preparatory study for the elaboration of a program
#              for autonomous cars.
# Date: 31/07/2022 by Antonio Castilho
# Ref.: The Complete Self-Driving Car Course - Applied Deep Learning
#       of https: // www.udemy.com


import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slop, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slop)
    x2 = int((y2 - intercept) / slop)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slop = parameters[0]
        intercept = parameters[1]
        if slop < 0:
            left_fit.append((slop, intercept))
        else:
            right_fit.append((slop, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_shift_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_shift_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to gray scale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # It's reduces noise
    canny = cv2.Canny(blur, 50, 150)  # Identify edges
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    poligons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poligons, 255)
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image


# img = cv2.imread('test_image.jpg')
# lane_img = np.copy(img)  # make a copy in numpy array format
# canny_img = canny(lane_img)
# cropped_image = region_of_interest(canny_img)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_img, lines)
# line_image = display_lines(lane_img, averaged_lines)
# combo_image = cv2.addWeighted(lane_img, 0.8, line_image, 1, 1)
# cv2.imshow('Mask', combo_image)
# cv2.waitKey(0)

video = 'test2.mp4'
cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    fr_true, frame = cap.read()
    canny_img = canny(frame)
    cropped_image = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow(video, combo_image)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
