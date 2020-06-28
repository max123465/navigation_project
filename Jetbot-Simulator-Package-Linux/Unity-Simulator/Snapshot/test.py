import numpy as np
import cv2

frames = 0


objpoints = []
Matrix = np.array([[568.67291932, 0.00000000e+00, 518.70213251],
 [0.00000000e+00, 567.49287398, 245.11856484],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
k = np.array([[ -0.30992158, 0.10084567,  0.00088568 ,-0.00114713, -0.01561677]])

def detect_chessboard(image):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    objp = np.zeros((4*3,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)*20

    img = image
    h,w,c = img.shape
    bytesPerLine = 3*w
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (4,3),None) #
    if ret:
        objpoints.append(objp)
        print('find target%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return objp,corners
    else:
        print('cant find target')
        return 0,0


img = cv2.imread('2020-06-28-15_16_22.jpg')
detect_chessboard(img)
img = cv2.imread('2020-06-28-15_16_31.jpg')
detect_chessboard(img)
img = cv2.imread('2020-06-28-15_16_34.jpg')
detect_chessboard(img)
