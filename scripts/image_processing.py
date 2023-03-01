# IMPORTANT: run this in terminal before to activate opencv enviornment
# conda install -c conda-forge opencv 
# conda activate opencv-env

 
import numpy as np
import cv2 as cv
import glob
import os,sys 
from matplotlib import pyplot as plt
from pathlib import Path
 
path0 = '/Users/madisonforman/Desktop/processing/data/*.jpg'
#jewett scans is not currently in here, should be 44 images
# path1 = '/Users/madisonforman/Desktop/processing/data/jewettscans/*.jpg'
cur = 0
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
        # os.rename(os.path.join(dir, image), os.path.join(dir,filename))
    return names

def load_images(path):
    cv_imgs = []
    for img in glob.glob(path):
        i = cv.imread(img)
        cv_imgs.append(i)
    return cv_imgs
#alpha controls brightness, beta controls contrast
def preprocess_images(path, alpha, beta):
    cur = 0
    image_list = load_images(path)
    name_list = create_names(path)

    print(name_list)
    i = 0
    for image in image_list:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #1 grayscale and contrast
        # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        adjusted = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
        #2 blur and divide m 
        #dividing all the values by 255 will convert it to range from 0 to 1.
        blur = cv.GaussianBlur(adjusted, (0,0), sigmaX=33, sigmaY=33)
        divide = cv.divide(adjusted, blur, scale=225)
        gray_filtered = cv.inRange(divide, 0, 100)
        #3 binarization
        _, thresh = cv.threshold(adjusted, 200, 255, cv.THRESH_BINARY)
        normalized = cv.normalize(adjusted, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
        #4 noise removal
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        erosion = cv.erode(morph, kernel, iterations=1)
        #cv.imshow("thresh", thresh)
        # cv.imshow("morph", morph)
        # cv.imshow("og", image)
        # cv.imshow("gray filter", gray_filtered)
        # cv.imshow("normalized", normalized)
        cv.imshow("eroded", erosion)

        cv.waitKey()
        cv.destroyAllWindows()
        cur += 1
        if cur == 5:
            break
        # os.chdir('/Users/madisonforman/Desktop/processing/data')
        # cv.imwrite(name_list[i], morph)
        i += 1
preprocess_images(path0, 1, -10)
#preprocess_images(path1, 1, 30)



"""
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