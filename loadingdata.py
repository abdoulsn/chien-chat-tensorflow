import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

dataloc = "data/Petimages"
categories = ['Dog', 'Cat']

for category in categories:
    path = os.path.join(dataloc, category)