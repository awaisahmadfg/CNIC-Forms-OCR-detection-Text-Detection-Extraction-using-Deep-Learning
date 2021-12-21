######## Finding RGB value ###########
# import os, sys
# from PIL import Image
#
# im = Image.open("C:/Users/hp/Desktop/Img1.png")
# x = 3
# y = 4
#
# pix = im.load()
# print (pix[x,y])



# from PIL import Image
# import numpy as np
# import cv2
#
# image = Image.open('C:/Users/hp/Desktop/Img1.png')
# width, height = image.size
# pixval = list(image.getdata())
#
# temp = []
# hexcolpix = []
# for row in range(0, height, 1):
#     for col in range(0, width, 1):
#         index = row*width + col
#         temp.append(pixval[index])
#     hexcolpix.append(temp)
#     temp = []
#
# #print (hexcolpix)
#
# im = cv2.imread('C:/Users/hp/Desktop/Img1.png')
#
# blue = [255,255, 0]
#
# # Get X and Y coordinates of all blue pixels
# X,Y = np.where(np.all(im==blue,axis=2))
#
# print(X,Y)

import cv2
import numpy as np

## (1) Read and convert to HSV
img = cv2.imread("C:/Users/hp/Desktop/Img1.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
## (2) Find the target yellow color region in HSV
hsv_lower = (76,76,129,255)
hsv_upper = (150,150,191,255)



mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

## (3) morph-op to remove horizone lines
kernel = np.ones((5,1), np.uint8)
mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)


## (4) crop the region
ys, xs = np.nonzero(mask2)
ymin, ymax = ys.min(), ys.max()
xmin, xmax = xs.min(), xs.max()

croped = img[ymin:ymax, xmin:xmax]

pts = np.int32([[xmin, ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
cv2.drawContours(img, [pts], -1, (0,255,0), 1, cv2.LINE_AA)

cv2.imshow("croped", croped)
cv2.imshow("img", img)

image_gray = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/hp/Desktop/crop image/Img1.png', image_gray)
cv2.waitKey()