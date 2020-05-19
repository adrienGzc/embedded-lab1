import argparse
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def main(image):
    img = cv.imread(image)
    imgToGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgDst = cv.fastNlMeansDenoising(imgToGray)
    imgBlured = cv.GaussianBlur(imgDst, (5, 5), 1)
    
    # Laplacian filter
    laplacian = cv.Laplacian(imgBlured, cv.CV_64F)

    # Sobel filter
    grad_x = cv.Sobel(imgBlured, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(imgBlured, cv.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, gamma=0)

    canny = cv.Canny(imgBlured,100,200)
    plt.subplot(121), plt.imshow(canny, cmap='gray')
    plt.subplot(122), plt.imshow(grad, cmap='gray')
    plt.show()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="Image for edge detection")
    img = vars(args.parse_args())    
    
    main(img['image'])