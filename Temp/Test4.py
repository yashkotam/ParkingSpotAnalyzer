import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('parking2-min.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform a blackhat morphological operation that will allow
# us to reveal dark regions (i.e., text) on light backgrounds
# (i.e., the license plate itself)
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

#plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB))
#plt.show()

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
    dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
plt.imshow(cv2.cvtColor(gradX, cv2.COLOR_BGR2RGB))
# plt.show()


gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
# plt.show()

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
     cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]

plateImages = []

for c in cnts:
    rect = cv2.boundingRect(c)
    (x, y, w, h) = rect
    ar = w / float(h)
    if ar >= 3 and ar <= 7:
        plateImages.append( img[y-2:y+h+2,x-2:x+w+2] )

# out = cv2.drawContours(img, plates, -1,255, 5)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for ig in plateImages:
    plt.imshow(cv2.cvtColor(ig, cv2.COLOR_BGR2RGB))
    print(pytesseract.image_to_string(ig, lang='eng', config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    plt.show()
