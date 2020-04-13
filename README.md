# harris_corner_detector
This program implement the Harris algorithm for corner detection.
using the Sobel filter to extract the derivatives of the image in X axis and the Y axis.
then construct the tensor setup matrix and calculate the minimum eigenvalue for each set of coordinate at the image
if the minimum eigenvalue is above the threshold (300), mark the point (corner).

# Requirements
- python 3.7 (tested)
- numpy library
- scipy library
- cv2 library
- skimage

# Walk through
there is an image from google, just for the example.
you can set the threshold in the if statement on the code, and choose the color of the corners marked.
