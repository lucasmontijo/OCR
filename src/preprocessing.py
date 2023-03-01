import cv2
import pytesseract
import config

def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian(image, gaussian_a = config.GAUSSIAN_A, gaussian_b = config.GAUSSIAN_B):
    if (gaussian_a % 2 == 0) or (gaussian_b % 2 == 0):
        raise Exception("Config arguments must be odd.")
    return cv2.GaussianBlur(image, (gaussian_a, gaussian_b), 0)

def remove_noise(image, blur_strength = config.NOISE_REDUCTION_STRENGTH):
    return cv2.medianBlur(image,blur_strength)

def thresholding(image):
    return cv2.threshold(image, 0, 255, config.THRESHOLDING + cv2.THRESH_OTSU)[1]

def gaussian_thresholding(image):
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,config.NOISE_REDUCTION_STRENGTH)

def canny(image, thr1=100, thr2=200):
    return cv2.Canny(image, thr1, thr2)

def get_data(image):
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='por')

def get_boxes(image):
    try:
        h, w, c = image.shape
    except:
        h, w = image.shape
    boxes = pytesseract.image_to_boxes(image) 
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 255, 255), 2)
    return image

def get_word_boxes(image):
    d = get_data(image)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def erode(image, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.ERODE_X, config.ERODE_Y))
    return cv2.erode(image, kernel, iterations=iterations)

def dilate(image, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.DILATE_X, config.DILATE_Y))
    return cv2.dilate(image, kernel, iterations=iterations)

def get_string(image):
    return pytesseract.image_to_string(image = image, lang=config.LANG)

def get_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    return contours, hierarchy

def draw_contours(image, contours):
    return cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
    