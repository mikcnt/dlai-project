import cv2
import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread("image.png", 1)

# cv2.bitwise_or(rectangle, circle)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color1_position = [55, 694]
color2_position = [67, 697]

color1 = hsv[color1_position[0], color1_position[1], :]
color2 = hsv[color2_position[0], color2_position[1], :]

# print(color1, color2)

lower_brown = np.array([10, 100, 20])
upper_brown = np.array([20, 255, 200])

color1[1] = 0
color1[2] = 0

color2[1] = 255
color2[2] = 255

mask = cv2.inRange(hsv, lower_brown, upper_brown)


kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# result = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow("img", img)
cv2.imshow("mask", mask)

# cv2.imshow('mask', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
