"""
Script for collecting datasets from the keyboard
"""

from GetKeys import get_key
from Screen import get_screen

import cv2
import numpy as np
import Detect_car
import time

def keys_to_output(keys):
    output = 0
    if 'A' in keys:
        output = 1
    elif 'D' in keys:
        output = 2
    return output

def proccess_image(orig_img):
    image = Detect_car.main(orig_img)
    return image

X1 = 0
Y1 = 40
WEIGHT = 1280 + X1
HEIGHT = 720 + Y1

size = (X1, Y1, WEIGHT, HEIGHT)

data = []
paused = True

while(True):
    screen = get_screen(size)
    image = proccess_image(screen)

    if not paused:
        keys = get_key()
        output = keys_to_output(keys)
        data.append([image, output])
        if len(data) % 500 == 0:
            print(len(data))
    cv2.imshow('window', image)
    keys = get_key()
    if False: #'O' in keys:
        if paused:
            print('NOT Paused')
            paused = False
            time.sleep(1)
        else:
            print('Paused')
            paused = True
            time.sleep(1)
            
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
if data != []:
    ch = input('Save data ? (y/n): ')
    if ch == 'y':
        np.save('data.npy', data)