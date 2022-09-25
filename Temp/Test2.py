import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('parking1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plates = []
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour,10, True)
    if len(approx) == 4:
        plates.append(approx)

print(plates)

'''mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, plates, -1,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)'''

new_image = cv2.drawContours(img, plates, -1,255, 5)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

plt.show()
