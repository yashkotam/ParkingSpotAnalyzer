import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

img = cv2.imread('parking1-min.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform a blackhat morphological operation that will allow
# us to reveal dark regions (i.e., text) on light backgrounds
# (i.e., the license plate itself)
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB))
#plt.show()

# next, find regions in the image that are light
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
#plt.imshow(cv2.cvtColor(light, cv2.COLOR_BGR2RGB))
#plt.show()
light = cv2.threshold(light, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]   
#plt.imshow(cv2.cvtColor(light, cv2.COLOR_BGR2RGB))
#plt.show()

# compute the Scharr gradient representation of the blackhat
# image in the x-direction and then scale the result back to
# the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
    dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
plt.imshow(cv2.cvtColor(gradX, cv2.COLOR_BGR2RGB))
# plt.show()

# blur the gradient representation, applying a closing
# operation, and threshold the image using Otsu's method
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
# plt.show()

# perform a series of erosions and dilations to clean up the
# thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))

# take the bitwise AND between the threshold result and the
# light regions of the image
'''thresh = cv2.bitwise_and(thresh, thresh, mask=light)'''
#thresh = cv2.dilate(thresh, None, iterations=2)
#thresh = cv2.erode(thresh, None, iterations=1)

plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.show()

# find contours in the thresholded image and sort them by
# their size in descending order, keeping only the largest
# ones
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

plates = []
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.04 * peri, True)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        # print(x)
        ar = w / float(h)
        if ar >= 3 and ar <= 7:
            print(c)
            plates.append(c)        

out = cv2.drawContours(img, plates, -1,255, 5)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))    
plt.show()