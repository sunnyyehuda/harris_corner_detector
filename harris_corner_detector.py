import cv2
from skimage.color import rgb2gray
from scipy import signal
import numpy as np


def harris_corner_detection(image):
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
    for y in range(image.shape[1]-1):
        for x in range(image.shape[0]-1):
            Sxx = np.sum(Ixx[x:x + 2, y:y + 2])
            Syy = np.sum(Iyy[x:x + 2, y:y + 2])
            Sxy = np.sum(Ixy[x:x + 2, y:y + 2])
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            min_eigenvalue = det/trace
            # set the threshold for the minimum eigenvalue as high as 200 for detect only the sharpest corners
            if min_eigenvalue > 300:
                corners.append((x, y))
    return corners


if __name__ == '__main__':
    img = cv2.imread('vanishing.png')
    cv2.imshow('original image', img)
    corners = harris_corner_detection(img)
    for x, y in corners:
        img[x, y] = (0, 255, 0)
    cv2.imshow('corner detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
