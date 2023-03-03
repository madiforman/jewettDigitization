# IMPORTANT: run this in terminal before to activate opencv enviornment
# conda install -c conda-forge opencv (only needs to be done once)
# conda activate opencv-env

 
import numpy as np
import cv2 as cv
import glob
 
path0 = '/Users/seung/Documents/Spring23/jewettDigitization/data/*.jpg' 
cur = 0
def create_names(path):
    lang = 'eng'
    font = 'jewett'
    str = f"{lang}.{font}.exp"
    names = []
    i = 0
    for img in glob.glob(path):
        filename = f"{str}{i}.jpg"
        names.append(filename)
        i += 1
    return names

def load_images(path):
    cv_imgs = []
    for img in glob.glob(path):
        i = cv.imread(img)
        cv_imgs.append(i)
    return cv_imgs

def remove_horizontal_lines(image):

    thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 1))
    image1 = cv.dilate(thresh, horizontal_kernel, iterations=1) 

    detected_lines = cv.morphologyEx(image1, cv.MORPH_OPEN, horizontal_kernel, iterations = 2)
    cv.imshow('lines', detected_lines)

    contours = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        cv.drawContours(image, [c], -1, (255, 255, 255), 2)

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

def remove_lines(image):
    image = remove_horizontal(image)
    edges = cv.Canny(image, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=50, maxLineGap = 150)
    for line in lines:
        x0, y0, x1, y1 = line[0]
        cv.line(image, (x0, y0), (x1, y1), (255, 0, 0), 3)
    cv.imshow("remove lines", image)

    repair_image(image)

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
    i = 0
    for image in image_list:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        adjusted = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
        remove_lines(gray)
        other = remove_horizontal(gray)

        cv.waitKey()
        cv.destroyAllWindows()
        cur += 1
        if cur == 10:
            break
        i += 1

preprocess_images(path0, 1, -10)


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