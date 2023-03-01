import numpy as np
import cv2 as cv
import glob
import os,sys 
from matplotlib import pyplot as plt

path0 = '/content/drive/MyDrive/repo/jewettDigitization/data'
path1 = '/content/drive/MyDrive/repo/jewettDigitization/jewettscans'

os.chdir(path0)
num_files = len(os.listdir('./'))

for i in range(num_files):
    os.system(f"tesseract english.jewett.exp{i}.jpg english.jewett.exp{i} batch.nochop makebox")