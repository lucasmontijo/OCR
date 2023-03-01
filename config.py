import cv2

#Arquivo
PATH = 'data/images/'
FILE = 'ipp_noisy_70'
FORMAT = '.jpg'

#Língua
LANG = 'por'

#Processamento
NOISE_REDUCTION_STRENGTH = 9

GAUSSIAN_A = 21
GAUSSIAN_B = 21

#thresholding method
THRESHOLDING = cv2.THRESH_BINARY_INV

#ERODE and DILATE sizes (pixels)
ERODE_X = 2
ERODE_Y = 2
DILATE_X = 3
DILATE_Y = 3