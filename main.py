import argparse
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def nonMaximumSupression(image, angle):
    # Get image shape.
    y, x = image.shape
    # Create new image based on the shape of the input image.
    newImage = np.zeros((y, x))

    # Loop through Y abscisse. 
    for indexY in range(0, y - 1):
        # Loop through X abscisse.
        for indexX in range(0, x - 1):
            # Horizontal 0
            if (0 <= angle[indexY, indexX] < 22.5) or \
            (157.5 <= angle[indexY, indexX] <= 180) or \
            (-22.5 <= angle[indexY, indexX] < 0) or \
            (-180 <= angle[indexY, indexX] < -157.5):
                b = image[indexY, indexX + 1]
                c = image[indexY, indexX - 1]
            # Diagonal 45
            elif (22.5 <= angle[indexY, indexX] < 67.5) or (-157.5 <= angle[indexY, indexX] < -112.5):
                b = image[indexY + 1, indexX + 1]
                c = image[indexY - 1, indexX - 1]
            # Vertical 90
            elif (67.5 <= angle[indexY, indexX] < 112.5) or (-112.5 <= angle[indexY, indexX] < -67.5):
                b = image[indexY + 1, indexX]
                c = image[indexY - 1, indexX]
            # Diagonal 135
            elif (112.5 <= angle[indexY, indexX] < 157.5) or (-67.5 <= angle[indexY, indexX] < -22.5):
                b = image[indexY + 1, indexX - 1]
                c = image[indexY - 1, indexX + 1]           
                
            # Non-max Suppression
            if (image[indexY, indexX] >= b) and (image[indexY, indexX] >= c):
                newImage[indexY, indexX] = image[indexY, indexX]
            else:
                newImage[indexY, indexX] = 0
    return newImage

def sobelFilter(image):
    filterX = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    absoluteX = cv.convertScaleAbs(filterX)

    filterY = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    absoluteY = cv.convertScaleAbs(filterY)

    filteredImage = np.hypot(absoluteX, absoluteY)
    filteredImage = filteredImage / filteredImage.max() * 255
    angle = np.rad2deg(np.arctan2(filterY, filterX))

    return (filteredImage, angle)

def hysteresisThresholding(image, weak=10, strong=70):
    # Get image shape.
    y, x = image.shape
    # Create a new image and initial it at 0 everywhere based on the image input shape.
    finalImage = np.zeros((y, x))

    # Thanks to numpy, get pixel lower than my weak threshold point.
    weakX, weakY = np.where(image < weak)
    # Same as above but other way around, get pixel higher than my strong threshold point.
    strongPixelX, strongPixelY = np.where(image >= strong)
    # Get pixel in between, this is gonna be my weak pixels to improve.
    mixedX, mixedY = np.where((image <= strong) & (image >= weak))
    
    # Standardization of weak, strong, and my weak pixels to the same color.
    finalImage[weakX, weakY ] = 0
    finalImage[strongPixelX, strongPixelY] = 255
    finalImage[mixedX, mixedY] = 75

    # Loop through the Y abscisse.
    for indexY in range(0, y):
        # Loop through the X abscisse.
        for indexX in range(0, x):
            # If the point is a mixed point (in-between).
            if (finalImage[indexY, indexX] == 75):
                # Check all around the point if the pixel is a strong one (white), then add a white pixel into my final image.
                if 255 in [finalImage[indexY + 1, indexX - 1],
                finalImage[indexY + 1, indexX],
                finalImage[indexY + 1, indexX + 1],
                finalImage[indexY, indexX - 1],
                finalImage[indexY, indexX + 1],
                finalImage[indexY - 1, indexX - 1],
                finalImage[indexY - 1, indexX],
                finalImage[indexY - 1, indexX + 1]]:
                    finalImage[indexY, indexX] = 255
                # Otherwise put the pixel in black.
                else:
                    finalImage[indexY, indexX] = 0
    # Return the result after hysterisis thresholding.
    return finalImage

def main(image):
    # Read the image using Opencv library.
    img = cv.imread(image)
    
    # Conver the image in black and night with a grayscale.
    imgToGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Denoise the image from potentiel noise.
    imgDst = cv.fastNlMeansDenoising(imgToGray)
    
    # Blur the image using a Gaussian filter of 5 x 5.
    imgBlured = cv.GaussianBlur(imgDst, (5, 5), 0)
    
    # Filter the imageto get the edges.
    imgSobel, angle = sobelFilter(imgBlured)
    
    # Thin the edges with a nms filter.
    imgNMS = nonMaximumSupression(imgSobel, angle)

    # Use a hysteris threshold to transform weak pixel into strong one if needed.
    # tmp, weak, strong = threshold(imgNMS)
    # final = hysteresis(tmp, weak, strong)
    final = hysteresisThresholding(imgNMS)
    
    # Display the image at the beginning and after edge detection.
    plt.subplot(121), plt.imshow(imgToGray, cmap='gray')
    plt.subplot(122), plt.imshow(final, cmap='gray')
    plt.show()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="Image for edge detection")
    img = vars(args.parse_args())    
    
    main(img['image'])