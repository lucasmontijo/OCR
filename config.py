import cv2

#Arquivo
PATH = 'data/images/'
FILE = 'graphic'
FORMAT = '.jpg'

#Língua
LANG = 'por'

#Customização de parâmetros
#CUSTOM_CONFIG = r'-c tessedit_char_whitelist=qwertyuiopasdfghjklçzxcvbnmQWERTYUIOPASDFGHJKLÇZXCVBNM --psm 6'

#Processamento
NOISE_REDUCTION_STRENGTH = 9

GAUSSIAN_A = 15
GAUSSIAN_B = 15

#thresholding method
THRESHOLDING = cv2.THRESH_BINARY

#ERODE and DILATE sizes (pixels)
ERODE_X = 2
ERODE_Y = 2
DILATE_X = 2
DILATE_Y = 2
