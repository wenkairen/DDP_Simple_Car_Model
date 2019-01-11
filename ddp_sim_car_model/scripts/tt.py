import numpy as np
import matplotlib.pyplot as plt
import re
import yaml
import matplotlib.image as mpimg
import cv2


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



# image = read_pgm("mymap.pgm", byteorder='<')
# print("image:", image.shape)
x = -30
y = -30
res = 0.025

cx = int(x / res)
cy = int(y /res)

img = cv2.imread("mymap.pgm")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = 205
t = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
print(t.shape)
# print(t[2098,1535])
# print(t[1124,1515])
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.namedWindow("output1", cv2.WINDOW_NORMAL)
# t= cv2.circle(t, (cx, cy), 2, (255,0,0), 3)
cv2.imshow("output",t)
kernel = np.ones((5,5),np.uint8)
erode = cv2.erode(t,kernel,iterations = 1)
cv2.imshow("output1",erode)
cv2.waitKey(0)
# plt.imshow(t)
# plt.show()

def process_map(x, y, res, map):
    ## center point 
    img = cv2.imread("map")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cx = x / res
    cy = y /res
    retval, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

