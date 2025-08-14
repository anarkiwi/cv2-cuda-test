#!/usr/bin/env python3

import cv2
import numpy as np
import time

images = 1000
height = 4096
width = 4096

rand_images = [
    np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    for _ in range(images)
]
cv2.cuda.setDevice(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
s = time.time()
for image in rand_images:
    clahe.apply(image)
print(time.time() - s)

clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
src = cv2.cuda_GpuMat(height, width, cv2.CV_8UC1)
dst = cv2.cuda_GpuMat(height, width, cv2.CV_8UC1)
host = np.empty((height, width), np.uint8)
stream = cv2.cuda_Stream()
s = time.time()
for image in rand_images:
    src.upload(image, stream)
    clahe.apply(src, stream, dst)
    dst.download(stream, host)
    stream.waitForCompletion()
print(time.time() - s)
