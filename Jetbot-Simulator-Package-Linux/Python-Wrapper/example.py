from jetbotSim import Robot, Camera
import numpy as np
import cv2,math

frames = 0


objpoints = []
Matrix = np.array([[568.67291932, 0.00000000e+00, 518.70213251],
 [0.00000000e+00, 567.49287398, 245.11856484],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
k = np.array([[ -0.30992158, 0.10084567,  0.00088568 ,-0.00114713, -0.01561677]])

def detect_chessboard(image):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    objp = np.zeros((3*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:3].T.reshape(-1,2)*20

    img = image
    h,w,c = img.shape
    bytesPerLine = 3*w
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,3),None) #
    if ret:
        objpoints.append(objp)
        print(corners)
        print(objp)
        print('find target%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return ret,objp,corners
    else:
        print('cant find target')
        return ret,0,0
        
def solve_dis(objectPoints,imagePoints,cameraMatrix,distCoeffs):
    ret,R, T = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    return R, T





def execute(change):
    global robot, frames
    print("\rFrames", frames, end="")
    frames += 1


    # Visualize
    img = cv2.resize(change["new"],(640,360))
    cv2.imshow("camera", img)
    ret,objp,corners = detect_chessboard(img)
    if ret !=0:
        R, T = solve_dis(objp,corners,Matrix,k)
        print(R)
        print(T)
        print(math.sqrt(T[0]**2+T[1]**2+T[2]**2))
    

robot = Robot()
camera = Camera()
camera.observe(execute)
