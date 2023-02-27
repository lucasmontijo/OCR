import matplotlib.pyplot as plt
import cv2 as cv
import config

def show(image, min=10, max=10):
    plt.figure(figsize=(min,max))
    plt.imshow(image, cmap='gray')
    plt.show()

def read_image(path = config.PATH, file = config.FILE, format = config.FORMAT):
    return cv.imread(path + file + format)