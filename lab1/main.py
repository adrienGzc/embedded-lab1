import argparse
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

def main(image):
    img = cv.imread(image)
    # cv.imshow('Image', img)
    # imgToGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--image", required=True, help="Image for edge detection")
    img = vars(args.parse_args())    
    
    main(img['image'])