from collections import Counter
import re
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from wand.image import Image as WandImage

lane_width = 146.5
hole_to_end_min = 96.5
hole_to_end_max = 107.5
hole_to_end = (hole_to_end_min + hole_to_end_max)/2
correct_ratio = lane_width / hole_to_end

sc = 640/2704
x_grid_corrected = np.array([285*sc, 1254*sc])
y_grid_corrected = np.array([360*sc, 1035*sc, 1710*sc])

hole = [x_grid_corrected[1], y_grid_corrected[1]]

ydiff = (y_grid_corrected[1] - y_grid_corrected[0])
xdiff = (x_grid_corrected[1] - x_grid_corrected[0])

pix_to_m = lane_width / 100 / (ydiff*2)

green_lower = np.array([30, 0, 0]) # green
green_upper = np.array([80, 255, 255]) 

def undistort_fixed(img_,h=480,w=640,c=3):
    '''
    manually tuned undistort function for golfputting study
    '''
    barr0=-.29
    barr1=1.2
    A=0.01
    B=0.0
    rot=1.
    shift_y=13
    shift_x=-26
    
    w_scale = 1.085

    h_mid = h//2 + shift_y
    w_mid = w//2 + shift_x
    image_center = tuple([w_mid, h_mid])
    rot_mat = cv2.getRotationMatrix2D(image_center, rot, 1.0)
    img_rot = cv2.warpAffine(img_, rot_mat, img_.shape[1::-1], flags=cv2.INTER_LINEAR)

    img_wand = WandImage.from_array(img_rot)
    img_wand.distort('barrel', (A, B, barr0, barr1, w_mid, h_mid))
    
    img_rescale = cv2.resize(np.array(img_wand), [int(w*w_scale), h])
    return img_rescale

def find_circles_in_img(img, plot_it=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, green_lower, green_upper)

    img2 = img.copy()
    img2[mask == 0] = 0
    img2[mask != 0] = 125
    
    kernel = np.ones((4,4),np.uint8)
    img2 = cv2.erode(img2, kernel, iterations=8)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))    
    
    _, cc = cv2.connectedComponents(gray_blurred)
    cnt = Counter(cc.ravel()).most_common(2)
    winner = cnt[0][0]
    if winner == 0:
        winner = cnt[1][0]
            
    gray_blurred[cc!=winner]=0
    
    detected_circles = cv2.HoughCircles(gray_blurred, 
                       cv2.HOUGH_GRADIENT, 3, 40, param1=20,
                       param2=50,minRadius = 5, maxRadius = 80)

    if plot_it:
        plt.imshow(mask)
        plt.figure()
        plt.imshow(img2)
        plt.imshow(gray_blurred)
        plt.figure()
        if detected_circles is not None:
            print(detected_circles.shape)
        # Draw circles that are detected.
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 4)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 6)
        plt.imshow(img)
    
    if detected_circles is not None:
        return detected_circles[0]

def find_circles_in_img_no_hough(img, plot_it=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # find the golf range surface
    mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # open and close the mask to remove noise
    kernel = cv2.getGaussianKernel(4, 0.6)
    img2 = cv2.erode(mask, kernel, iterations=6)
    img2 = cv2.dilate(img2, kernel, iterations=6)
    
    # remove all but the biggest green spot
    _, cc = cv2.connectedComponents(img2)
    cnt = Counter(cc.ravel()).most_common(2)
    winner = cnt[0][0]
    if winner == 0:
        winner = cnt[1][0]   
    img2[cc != winner] = 0
    
    # invert and find holes in green (must be ball and holes)
    n2, cc2 = cv2.connectedComponents(cv2.bitwise_not(img2))
    cnt2 = Counter(cc2.ravel()).most_common(n2)
    # the biggest components are background and lane, take 3 onwards
    cands = cnt2[2:]

    centers = []
    for c in cands:
        dots = np.where(cc2==c[0])
        cent = np.mean(dots, 1)
        s = np.sum(cc2==c[0])
                
        if s > 100 and s < 10000:
            centers.append(cent[::-1])
    centers = np.array(centers)

    if plot_it:
        print('cc2:', n2, np.unique(cc2))
        plt.imshow(mask)
        plt.figure()
        plt.imshow(img2)
        plt.figure()
        plt.imshow(cc2)
        plt.figure()
        plt.imshow(img)
        plt.scatter(*centers.T, c='red', s=10)
    return centers

# find if there is a card in the image.
def find_cards(img_straight_, plot_it=False):
    hsv = cv2.cvtColor(img_straight_, cv2.COLOR_RGB2HSV)
    hsl = cv2.cvtColor(img_straight_, cv2.COLOR_RGB2HLS)

    # blue
    lower = np.array([110, 50, 50])
    upper = np.array([134, 255, 255])
    maskb = cv2.inRange(hsv, lower, upper)
    blue_val = np.sqrt(np.sum(maskb!=0))
        
    # white
    lower = np.array([0, 150, 0])
    upper = np.array([255, 255, 255]) 
    maskw = cv2.inRange(hsl, lower, upper)
    maskw[np.sum(img_straight_, 2)==0] = 0
    
    white_val = np.sqrt(np.sum(maskw!=0))
    
    if plot_it:
        plt.figure()
        plt.imshow(maskb)
        plt.figure()
        plt.imshow(maskw)
        
    
    return blue_val, white_val

crop_area = [int(450*sc), int(1700*sc), int(350*sc)]
def filter_img_by_green(img, crop_area=crop_area):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # find the golf range surface
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # open and close the mask to remove noise
    kernel = cv2.getGaussianKernel(4, 0.6)
    mask2 = cv2.erode(mask, kernel, iterations=6)
    mask2 = cv2.dilate(mask2, kernel, iterations=6)

    # remove all but the biggest green spot
    _, cc = cv2.connectedComponents(mask2)
    cnt = Counter(cc.ravel()).most_common(2)
    winner = cnt[0][0]
    if winner == 0:
        winner = cnt[1][0]
    mask2[cc != winner] = 0
    
    # remove everything outside of the green
    contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask3 = np.zeros(mask2.shape, np.uint8)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img3 = cv2.drawContours(mask3, contours, 0, (255), -1)
    filtered_img = img.copy()
    filtered_img[img3 == 0] = 0
    green_points = np.sqrt(np.sum(img3/255))
    
    # we didnt actually find the green. abort!
    if green_points < 1500:
        filtered_img = img
        
    # additionally crop out area that is def outside of green
    if crop_area is not None:
        filtered_img[:crop_area[0], :] = 0
        filtered_img[crop_area[1]:, :] = 0
        if crop_area[2] < 0:
            filtered_img[:, crop_area[2]:] = 0
        else:
            filtered_img[:, :crop_area[2]] = 0
    return filtered_img
    
def key_to_res_name(k, prefix='P'):
    fileparts = k.split('/')
    reg = '^%s[0-9]+'%prefix
    for fi, f in enumerate(fileparts):
        res = re.match(reg, f)
        if res:
            pname = f.split(' ')[0]
            exp_name = '-'.join(fileparts[fi+1:]).replace(' ', '-')
            break

    return '_'.join([pname, exp_name])

def get_red_hole(img_col_):
    hsv = cv2.cvtColor(img_col_, cv2.COLOR_RGB2HSV)
    # red
    lower = np.array([15, 0, 0])
    upper = np.array([165, 255, 255])
    maskr = 255 - cv2.inRange(hsv, lower, upper)
    red_val = np.sqrt(np.sum(maskr!=0))
    maskr[:, :1500] = 0
    maskr[:, 3000:] = 0
    maskr[:600] = 0
    maskr[2200:] = 0

    n_red, cc = cv2.connectedComponents(maskr)
    if n_red == 2:
        hole = np.mean(np.where(cc==1), 1)[::-1]
        return hole
