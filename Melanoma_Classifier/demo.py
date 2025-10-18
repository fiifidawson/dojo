import cv2
import numpy as np
import torch
from net_class import Net

def apply_model(path):

    # 50x50 pixels
    img_size = 50

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    img_array = np.array(img)