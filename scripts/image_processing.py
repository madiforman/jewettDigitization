# IMPORTANT: run this in terminal before to activate opencv enviornment
# conda install -c conda-forge opencv (only needs to be done once)
# conda activate opencv-env

 
import numpy as np
import cv2 as cv
import glob
import os,sys 
from matplotlib import pyplot as plt
from pathlib import Path
 
path0 = '/Users/madisonforman/Desktop/jewettDigitization/data/*.jpg' 
#jewett scans is not currently in here, should be 44 images
# path1 = '/Users/madisonforman/Desktop/processing/data/jewettscans/*.jpg'
cur = 0
#creates string names for tesseract syntax
def create_names(path):
    # images = [f for f in os.listdir(dir)]
    lang = 'eng'

    font = 'jewett'
    str = f"{lang}.{font}.exp"
    names = []
    i = 0
    for img in glob.glob(path):
        filename = f"{str}{i}.jpg"
        names.append(filename)
        i += 1
        # os.rename(os.path.join(dir, image), os.path.join(dir,filename)) #files have already been named so dont run this rn
    return names
#loads images as cv images
def load_images(path):
    cv_imgs = []
    for img in glob.glob(path):
        i = cv.imread(img)
        cv_imgs.append(i)
    return cv_imgs
# def connect_lines(image):
def remove_horizontal_lines(image):
    # cv.imshow('src image', image)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 1))
    image1 = cv.dilate(thresh, horizontal_kernel, iterations=1) 

    detected_lines = cv.morphologyEx(image1, cv.MORPH_OPEN, horizontal_kernel, iterations = 2)
    cv.imshow('lines', detected_lines)

    contours = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # cv.imshow('lines', detected_lines)
    for c in contours:
        cv.drawContours(image, [c], -1, (255, 255, 255), 2)
    # cv.imshow(erosion, erosion)
    return image
def remove_horizontal(image):
    h = float(image.shape[0])
    maxVal = 250
    blockSize = 15
    C = 12.0*(90.0/h)
    bw = cv.adaptiveThreshold(image, maxVal, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, C)
    bw = ~bw
    vertical = bw.copy()
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    vertical = cv.erode(vertical, verticalStructure, None, (-1,-1))
    vertical = cv.dilate(vertical, verticalStructure, None, (-1,-1))
    vertical = ~vertical
    cv.imshow("remove_horiz", vertical)
    return vertical
def remove_noise(image):
    blur = cv.GaussianBlur(image,(13,13),0)
    thresh = cv.threshold(blur, 100,255, cv.THRESH_BINARY)[1]
    cv.imshow("final 2", thresh)
def repair_image(image):
    ret, thresh1 = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = np.array((5, 2), np.uint8)
    closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    remove_noise(image)
    # cv.imshow("final", closing)
def remove_lines(image):
    image = remove_horizontal(image)
    edges = cv.Canny(image, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=50, maxLineGap = 150)
    for line in lines:
        x0, y0, x1, y1 = line[0]
        cv.line(image, (x0, y0), (x1, y1), (255, 0, 0), 3)
    cv.imshow("remove lines", image)
    # image = remove_horizontal(image)
    repair_image(image)
    # cv.imshow("result", image)

    # cv.imshow("result", image)

# def remove_linesII(image):
#     ker0 = np.ones((3,5), np.uint8)
#     ker1 = np.ones((9,9), np.uint8)

#     imBW = cv.threshold(image, 230, 255, cv.THRESH_BINARY_INV)[1]
#     image0 = cv.erode(imBW, ker0, iterations = 1)
#     image1 = cv.dilate(image0, ker1, iterations=3)
#     image2 = cv.bitwise_and(imBW, image1)
#     image2 = cv.bitwise_not(image2)
#     image3 = cv.bitwise_and(imBW, imBW, mask=image2)
#     lines = cv.HoughLinesP(image3, 15, np.pi/180, 10, minLineLength = 440, maxLineGap = 15)
    
#     for i in range(len(lines)):
#         for x0, y0, x1, y1 in lines[i]:
#             cv.line(image, (x0, y0), (x1, y1), (0,255,0),2)
#     return image

"""
Preprocess images does the image preprocessing
- alpha controls brightness, beta controls contrast
- alpha range: 0 < alpha < 1
- beta range:  [-127, 127]
""" 
def preprocess_images(path, alpha, beta):
    cur = 0
    image_list = load_images(path)
    name_list = create_names(path)
    # print(name_list)
    i = 0
    for image in image_list:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #1 grayscale and contrast
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        adjusted = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
        remove_lines(gray)
        other = remove_horizontal(gray)
        # cv.imshow("other", other)
        #2 blur and divide 
        #dividing all the values by 255 will convert it to range from 0 to 1.
        # blur = cv.GaussianBlur(adjusted, (0,0), sigmaX=33, sigmaY=33)
        # divide = cv.divide(adjusted, blur, scale=225)
        # gray_filtered = cv.inRange(divide, 0, 100)

        # #3 binarization
        # _, thresh = cv.threshold(gray_filtered, 200, 255, cv.THRESH_BINARY)
        # normalized = cv.normalize(gray_filtered, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    
        # #4 noise removal
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        # morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        # erosion = cv.erode(morph, kernel, iterations=1)
        
        # cv.imshow("dilated", dilated)
        # cv.imshow("thresh", thresh)
        # cv.imshow("morph", morph)
        # cv.imshow("og", image)
        # cv.imshow("gray filter", gray_filtered)
        # cv.imshow("normalized", normalized)
        # cv.imshow("eroded", erosion)

        cv.waitKey()
        cv.destroyAllWindows()
        cur += 1
        if cur == 10:
            break
        # os.chdir('/Users/madisonforman/Desktop/processing/data')
        # cv.imwrite(name_list[i], morph)
        i += 1
preprocess_images(path0, 1, -10)
#preprocess_images(path1, 1, 30)



"""
Notes from meeting with chambers:
one approach: pretraining: train classifier on print, signatures, etc 
supplement the rest of training with transcriptions of Jewett's notes
maybe look for a pretrained OCR, or create a pretrained model
one person working on using a pretrained OCR, the other building it from scratch
find something trained on Bentham
Results by the midterm
tesseract??
THE MORE DATA THE BETTER
(hold some back as test data)
- transcribe a page and calculate how much time it would take to transcibe 500 pages
step 3: putting back on the web 
Measuring accuaracy 
second approach: 
find a premade ocr that gets higher accuracy, and correct that instead
have one/two person scanning and transcribing
other two people find a dataset as close as possible and begin training that
"""
# FOLLOWING THIS LINE FUNCTIONS HAVE NOT BEEN WORKING SUPER WELL <3 MIGHT BE BETTER ON SEGMENTED IMAGES
#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
def getSkewAngle(cvImage):
    new = cvImage.copy()
    gray = cv.cvtColor(new, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=5)

    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    
    largest_countour = contours[0]
    min_area_rect = cv.minAreaRect(largest_countour)
    angle = min_area_rect[-1]
    if angle < -45:
        angle += 90
    return -1.0 * angle
def rotate_image(cvImage, angle):
    new = cvImage.copy()
    (h,w) = new.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    new = cv.warpAffine(new, M, (w,h),
    flags = cv.INTER_CUBIC, borderMode = cv.BORDER_REPLICATE)
    return new

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotate_image(cvImage, -1.0 * angle)

def remove_borders(cvImage):
    new = cvImage.copy()
    mask = np.zeros(new.shape, dtype=new.dtype)
    _, thresh = cv.threshold(new, 205, 255, cv.THRESH_BINARY)
    cont = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]

    for c in cont:
        area = cv.contourArea(c)
        if area < 500:
            cv.drawContours(mask, [c], -1, (255, 255, 255), -1)
    removed_boarder = cv.bitwise_and(new, new, mask=mask)
    return removed_boarder



# # def remove_vertical_lines(image):
# #     thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
# #     vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))
# #     detected_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations = 2)
# #     contours = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# #     contours = contours[0] if len(contours) == 2 else contours[1]
# #     for c in contours:
# #         cv.drawContours(image, [c], -1, (255, 255, 255), 2)

#     return image