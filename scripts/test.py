from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2 as cv
import glob
from skimage.filters import sobel
from skimage.filters import threshold_otsu
import numpy as np
from heapq import *


path0 = '/Users/seung/Documents/Spring23/jewettDigitization/img.jpg' 

def load_images(path):
    cv_imgs = []
    for img in glob.glob(path):
        i = cv.imread(img)
        cv_imgs.append(i)
    return cv_imgs

def horizontal_projections(image):
    #threshold the image.
    sum_of_rows = []
    for row in range(image.shape[0]-1):
        sum_of_rows.append(np.sum(image[row,:]))
    
    return sum_of_rows

def find_peak_regions(hpp, threshold):
    peaks = []
    for i, hppv in enumerate(hpp):
        # print("threshold",threshold)
        # print("hppv",hppv)
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks

# def get_road_block_regions(nmap):
#     road_blocks = []
#     needtobreak = False
    
#     for col in range(nmap.shape[1]):
#         start = col
#         end = col+20
#         if end > nmap.shape[1]-1:
#             end = nmap.shape[1]-1
#             needtobreak = True

#         if path_exists(nmap[:, start:end]) == False:
#             road_blocks.append(col)

#         if needtobreak == True:
#             break
            
#     return road_blocks

def preprocess_images(path):
    
    image = imread(path0)
    sobel_image = sobel(image)
    
    hpp = horizontal_projections(sobel_image)

    threshold = (np.max(hpp)-np.min(hpp))/8
    peaks = find_peak_regions(hpp, threshold)
    peaks_indexes = np.array(peaks)[:, 0].astype(int)
    
    segmented_img = np.copy(image)

    r, c= segmented_img.shape

    for ri in range(r):
        if ri in peaks_indexes:
            segmented_img[ri, :] = 0
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,7))
    ax1.plot(hpp)
    ax2.set_title("threshold line")
    ax1.plot([0, image.shape[0]], [threshold, threshold,])
    ax2.imshow(segmented_img, cmap="gray")
    
    # group the peaks through which we will be doing path planning.
    diff_between_consec_numbers = np.diff(peaks_indexes) # difference between consecutive numbers
    indexes_with_larger_diff = np.where(diff_between_consec_numbers > 1)[0].flatten()
    peak_groups = np.split(peaks_indexes, indexes_with_larger_diff)
    
    # remove very small regions, these are basically errors in algorithm because of our threshold value
    peak_groups = [item for item in peak_groups if len(item) > 40]
    print("peak groups found", len(peak_groups))
    
    
    binary_image = get_binary(image)
    segment_separating_lines = []
    for i, sub_image_index in enumerate(peak_groups):
        nmap = binary_image[sub_image_index[0]:sub_image_index[-1]]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
        # print("thepath",len(path))
        offset_from_top = sub_image_index[0]
        if len(path) > 0:
            path[:,0] += offset_from_top
            segment_separating_lines.append(path)
    
    # visualize a sample
    cluster_of_interest = peak_groups[6]
    offset_from_top = cluster_of_interest[5]
    nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1],:]
    plt.figure(figsize=(5,5))
    plt.imshow(nmap, cmap="gray")
    path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
    plt.plot(path[:,1], path[:,0])
    plt.show()
    
    # draw lines onto the photo
    offset_from_top = cluster_of_interest[0]
    fig, ax = plt.subplots(figsize=(10,7), ncols=2)
    for path in segment_separating_lines:
        ax[1].plot((path[:,1]), path[:,0])
    ax[1].imshow(image, cmap="gray")
    ax[0].imshow(image, cmap="gray")
    plt.show()

def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if (array[neighbor[0]][neighbor[1]] == 1).any():
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return []

preprocess_images(path0)