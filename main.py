import cv2 as cv
import numpy as np
import time

start = time.time()
count = 0
cap = cv.VideoCapture(0)

# Creating black board
_, frame = cap.read()
board = np.zeros([frame.shape[0], frame.shape[1], 3], 'uint8')
board = cv.flip(board, 1)
while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 150, 100], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

    # Set range for blue color and define mask
    blue_lower = np.array([90, 150, 60], np.uint8)
    blue_upper = np.array([150, 255, 255], np.uint8)
    blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper)

    kernal = np.ones((5, 5), "uint8")

    red_mask = cv.dilate(red_mask, kernal)
    res_red = cv.bitwise_and(hsvFrame, hsvFrame, mask=red_mask)

    blue_mask = cv.dilate(blue_mask, kernal)
    res_blue = cv.bitwise_and(hsvFrame, hsvFrame, mask=blue_mask)

    # Creating contours
    # for red
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(i) for i in contours]
    if len(areas) > 0:
        i = areas.index(max(areas))
        if cv.contourArea(contours[i]) > 200:
            x, y, w, h = cv.boundingRect(contours[i])
            lefttop = x, y  # (x + w // 2, y + h // 2)
            board = cv.circle(board, lefttop, 5, (0, 0, 255), -1)
    # for contour in contours:
    #     area = cv.contourArea(contour)
    #     if area > 200:
    #         x, y, w, h = cv.boundingRect(contour)
    #         lefttop = x, y  # (x + w // 2, y + h // 2)
    #         board = cv.circle(board, lefttop, 5, (0, 0, 255), -1)
    # areas.append(area)
    # print(areas)
    # for blue
    contours, hierarchy = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 400:
            x, y, w, h = cv.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            board = cv.circle(board, center, 25, (0, 0, 0), -1)

    frame = cv.addWeighted(board, 1, frame, 1, 0)

    cv.imshow('result', frame)
    count += 1
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
end = time.time()
print(count / (end - start))
