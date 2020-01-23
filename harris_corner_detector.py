import cv2
from skimage.color import rgb2gray
from scipy import signal
import numpy as np


def harris_corner_detection(image, master_slave):
    corners = []
    image_gray = rgb2gray(image)
    image_smooth = signal.convolve2d(image_gray, np.array([[-1, -1, -1],
                                                      [-1, 32, -1],
                                                      [-1, -1, -1]]), mode='same')
    # Sobel_X filter
    Ix = signal.convolve2d(image_smooth, np.array([[-1, 0, 1],
                                                   [-2, 0, 2],
                                                   [-1, 0, 1]]), mode='same')
    # Sobel_Y filter
    Iy = signal.convolve2d(image_smooth, np.array([[1, 2, 1],
                                                   [0, 0, 0],
                                                   [-1, -2, -1]]), mode='same')
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2
    k = 0.03
    for y in range(int((image_gray.shape[1]-1)*(0.5*(1 - master_slave))),
                   int((image_gray.shape[1] - 1)*(1-master_slave*0.5))):
        for x in range(image_gray.shape[0]-1):
            Sxx = np.sum(Ixx[x:x + 2, y:y + 2])
            Syy = np.sum(Iyy[x:x + 2, y:y + 2])
            Sxy = np.sum(Ixy[x:x + 2, y:y + 2])
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k * (trace ** 2)
            if r > 0:
                corners.append((x, y))
    return corners


if __name__ == '__main__':
    img = cv2.imread('vanishing.png')
    cv2.imshow('original image', img)
    corners = harris_corner_detection(img, 0)
    for x, y in corners:
        img[x, y] = (0, 0, 255)
    cv2.imshow('corner detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
