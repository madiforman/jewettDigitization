# IMPORTANT: run this in terminal before to activate opencv enviornment
# conda install -c conda-forge opencv (only needs to be done once)
# conda activate opencv-env

 
import numpy as np
import cv2 as cv
import glob
import os,sys 
from matplotlib import pyplot as plt
from pathlib import Path
# import imutils

path0 = '/Users/seung/Documents/Spring23/jewettDigitization/data/*.jpg' 
#jewett scans is not currently in here, should be 44 images
# path1 = '/Users/madisonforman/Desktop/processing/data/jewettscans/*.jpg'
cur = 0
#creates string names for tesseract syntax
process_images = []
path_image = '/Users/seung/Documents/Spring23/jewettDigitization/data/1916fd-0006.jpg'
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
def remove_vertical(image):
    thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,50))
    detected_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=1)
    contours = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        cv.drawContours(image, [c], -1, (255,255,255), 2)
    return image
def remove_horizontal(image):
    edges = cv.Canny(image, 50, 150, apertureSize=3)

    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/100   #Angle resolution of the accumulator in radians.   
    threshold = 50        #Only lines that are greater than threshold will be returned.
    minLineLength = 750   #Line segments shorter than that are rejected.
    maxLineGap = 80    #Maximum allowed gap between points on the same line to link them
    color = (255, 255, 255)

    lines = cv.HoughLinesP(edges , rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)
    # image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            cv.line(image, (x0, y0), (x1, y1), color, 3)
    return image

  
def remove_noise(image):
    blur = cv.GaussianBlur(image,(7, 7),0)
    thresh = cv.threshold(image, 215  , 255, cv.THRESH_BINARY)[1]
    return thresh  

"""
Preprocess images does the image preprocessing
- alpha controls brightness, beta controls contrast
- alpha range: 0 < alpha < 1
- beta range:  [-127, 127]
"""
def preprocess_images(path, alpha, beta):
    cur = 0
    image_list = load_images(path)
    i = 0
    kernel = np.ones((4, 4), np.uint8)
    gray = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
    binarize =  cv.threshold(gray, 190, 255, cv.THRESH_BINARY)[1]
    image = remove_vertical(binarize)
    image = remove_horizontal(image)
    
    ret, thresh1 = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('thresh1', thresh1)
    
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, k)
    
    k1 = np.ones((3, 3), np.uint8)
    erosion = cv.erode(closing, k1, iterations = 1)
    # cv.imshow("erosion", erosion)
    # cv.imshow("image", image)        
    cv.waitKey()

    cv.imwrite("img.jpg", image)

preprocess_images(path_image, 1, -15)
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