import cv2 as cv
img = cv.imread("testi.png")

b = img[:, :, 0]  # Blue
g = img[:, :, 1]  # Green
r = img[:, :, 2]  # Red

cv.imshow("Display window", img)
cv.imshow("r", r)
cv.imshow("g", g)
cv.imshow("b", b)

k = cv.waitKey(0) # Wait for a keystroke in the window

1