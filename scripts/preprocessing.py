 #conda activate opencv-env #conda install -c conda-forge opencv
import numpy as np
import cv2 as cv
import json
from matplotlib import pyplot as plt
#CNN 

sigma = .10 
image = cv.imread('img034.jpg')
alpha = 3
beta = 0
# contrast = 110
# brightness = 0

adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# second = cv.addWeighted(image, contrast, image, 0,  brightness)
th, im_th = cv.threshold(adjusted, 200, 255, cv.THRESH_BINARY)


# blurred = cv.GaussianBlur(image, (7,7), 0)
# # ret, bw_img = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
# # cv.imshow("first", bw_img)
# cv.imshow("second", blurred)
normalized = cv.normalize(image, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
img = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)
noise_removed = cv.medianBlur(img, 3)

# th, im_th0 = cv.adaptiveThreshold(img, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,3,2)

window_name = 'image'
median = np.median(image)
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))
edge_image = cv.Canny(im_th, lower, upper)
# im_th0 = cv.adaptiveThreshold(edge_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,3,2)



# dst = cv.fastNlMeansDenoisingColored(edge_image, None, 10, 10, 7, 21)
# cv.imshow("three", dst)
cv.imshow("normalized", normalized)
cv.imshow("noise removed", noise_removed)

cv.imshow("one", edge_image)
# cv.imshow("edge + adaptive gaussian thresholding", im_th0)
cv.waitKey()
cv.destroyAllWindows()
# # # sharp = cv.filter2D(src=image, ddepth=-1, kernel=kernel)
# # # cv.imshow('AV CV- Winter Wonder Sharpened', sharp)


# # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# # divide = cv.GaussianBlur(image, (0,0), sigmaX=33, sigmaY=33)

# # threash = cv.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# # kernel = cv.getStructuringElement(cv2.MORPH_RECT, (3,3))
# # morph = cv.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# # cv.imshow("edgeDetection", edge_image)

# # ret, thresh_image = cv.threshold(edge_image, 0, 255, cv.THRESH_BINARY)


# cv.waitKey(0)
# cv.destroyAllWindows()
# # cv2.imwrite('img034.jpg', dst)