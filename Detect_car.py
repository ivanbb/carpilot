
"""
Script for car recognition
"""

import cv2
import numpy as np
import Model_detect_car as mdc
import skimage
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

OBLAST_W = [0, 200]
OBLAST_H = [0, 175]
WEIGHT = OBLAST_W[1] - OBLAST_W[0] #400
HEIGHT = OBLAST_H[1] - OBLAST_H[0] #350

heatmodel = mdc.create_model((HEIGHT, WEIGHT, 3))
mdc.load_model(heatmodel)
cv2.namedWindow('setting')

cv2.createTrackbar('tresh','setting', 0, 30, lambda x: x)
cv2.createTrackbar('box','setting', 0, 128, lambda x: x)

"""
	Select interesting region
"""
def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, [255,255,255])
	masked = cv2.bitwise_and(img, mask)
	return masked

"""
	Show value brightness for forward car
"""
def draw_text(text, img):
	font = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10, 10)
	fontScale = 0.3
	fontColor = (0,255,255)
	lineType = 1
	cv2.putText(img, str(text), bottomLeftCornerOfText, font,  fontScale, fontColor, lineType)
	return img

def draw_boxes(img, bboxes):
	draw_img = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)
	return draw_img

def search_cars(img):
	box = 64
	img = img.reshape(1,HEIGHT, WEIGHT,3)
	heat = heatmodel.predict(img)
	xx, yy = np.meshgrid(np.arange(heat.shape[2]),np.arange(heat.shape[1]))
	x = (xx[heat[0,:,:,0]>0.999])
	y = (yy[heat[0,:,:,0]>0.999])
	hot_windows = []
	for i,j in zip(x,y):
		hot_windows.append(((i*8,j*8), (i*8+box,j*8+box)))
	return hot_windows

def main(img):
	vertices = np.array([[0,HEIGHT],[90, 0], [141, 0], [WEIGHT,HEIGHT]], np.int32)
	orig_img = img
	img = img[OBLAST_H[0]:OBLAST_H[1], OBLAST_W[0]:OBLAST_W[1]]
	img = roi(img, [vertices])
	hot_windows = search_cars(img)
	window_img = draw_boxes(orig_img, hot_windows)
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	heat = add_heat(heat, hot_windows)
	heat = apply_threshold(heat, 3)
	heatmap = np.clip(heat, 0, 255)
	avg = np.average(heatmap)
	heatmap = np.uint8(heatmap)
	img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
	img = draw_text(avg, img)
	return window_img, img, avg

def add_heat(heatmap, bbox_list):
	tr = cv2.getTrackbarPos('tresh','setting')
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1 + 5
	return heatmap

def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap

def draw_labeled_bboxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
	return img