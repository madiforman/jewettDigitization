# IMPORTANT: run this in terminal before to activate opencv enviornment
# conda install -c conda-forge opencv (only needs to be done once)
# conda activate opencv-env

 
import numpy as np
import cv2 as cv
import glob
import os,sys 
from matplotlib import pyplot as plt
from pathlib import Path
import imutils

path0 = '/Users/madisonforman/Desktop/jewettDigitization/data/fieldDiary1916/*.jpg' 
#jewett scans is not currently in here, should be 44 images
# path1 = '/Users/madisonforman/Desktop/processing/data/jewettscans/*.jpg'
cur = 0
#creates string names for tesseract syntax
process_images = []
path_image = '/Users/madisonforman/Desktop/jewettDigitization/data/fieldDiary1916/1916fd-0037.jpg'
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
    cv.imshow("lines", image)
    cv.waitKey()
    return image

  
def remove_noise(image):
    # blur = cv.GaussianBlur(image,(7, 7),0)
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
    name_list = create_names(path)
    # print(name_list)
    i = 0
    template = image_list[0]
    for image in image_list:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binarize =  cv.threshold(gray, 215, 255, cv.THRESH_BINARY)[1]
        image = remove_vertical(binarize)
        image = remove_horizontal(image)
        cv.imshow("image", image)        
        cv.waitKey()
        cv.destroyAllWindows()
        cur += 1
        if cur == 10:
            break
        # os.chdir('/Users/madisonforman/Desktop/processing/data')
        # cv.imwrite(name_list[i], morph)
        i += 1
preprocess_images(path0, 1, -15)
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
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    #use ORB to detect keypoints and extract features
    orb = cv.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(gray, None)
    (kpsB, descsB) = orb.detectAndCompute(gray_template, None)
    #match features
    method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    #sort matches by distance (smaller distance => more similar)
    matches = sorted(matches, key=lambda x:x.distance)
    #keep only top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    #check to see if we shoudl visualize matched keypoints
    if debug:
        matchedVis = cv.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width = 1000)
        cv.imshow("Matched keypoints", matchedVis)
        cv.waitKey()
    #allocate mem for keypoints so we can comepute homography matrix
    ptsA = np.zeros((len(matches), 2), dtype='float')
    ptsB = np.zeros((len(matches), 2), dtype='float')
    #loop over top matches
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    #compute matrix between sets of matched points
    (H, mask) = cv.findHomography(ptsA, ptsB, method = cv.RANSAC)
    (h, w) = template.shape[:2]
    aligned = cv.warpPerspective(image, H, (w,h))
    #return aligned image
    cv.imshow("aligned", aligned)
    cv.waitKey()
    return aligned
# def preprocess_images(path, alpha, beta):
#     cur = 0
#     image_list = load_images(path)
#     name_list = create_names(path)
#     # print(name_list)
#     i = 0
#     template = image_list[0]
#     for image in image_list:
#         # image = align_images(image, template, maxFeatures = 500, keepPercent = 0.2, debug=True)
#         # straighten_image(image)
#         #1 grayscale and contrast
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         adjusted = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
#         #2 blur and divide 
#         #dividing all the values by 255 will convert it to range from 0 to 1.
#         # blur = cv.GaussianBlur(adjusted, (0,0), sigmaX=33, sigmaY=33)
#         # divide = cv.divide(adjusted, blur, scale=225)
#         # #3 binarization
#         # _, thresh = cv.threshold(divide, 0, 255, cv.THRESH_BINARY)
#         # normalized = cv.normalize(divide, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
   
#         remove_horizontal(test)
#         # #4 noise removal
#         # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
#         # morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
#         # erosion = cv.erode(morph, kernel, iterations=1)
        
#         # cv.imshow("dilated", dilated)
#         # cv.imshow("thresh", thresh)
#         # cv.imshow("morph", morph)
#         # cv.imshow("og", image)
#         # cv.imshow("gray filter", gray_filtered)
#         # cv.imshow("normalized", normalized)
#         # cv.imshow("eroded", erosion)

#         # cv.waitKey()
#         cv.destroyAllWindows()
#         cur += 1
#         if cur == 10:
#             break
#         # os.chdir('/Users/madisonforman/Desktop/processing/data')
#         # cv.imwrite(name_list[i], morph)
#         i += 1