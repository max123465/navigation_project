from jetbotSim import Robot, Camera
import numpy as np
import cv2
import os
import yaml

objpoints = []
frames = 0
v = 0.1
kp = 0.002
car_width = 0.8
# camera info path
yaml_dir = "config/"
camera_params = os.path.join(yaml_dir,"cam.yaml") 
with open(camera_params, "r") as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
K = np.array(data["K"])
D = np.array(data["D"])
proj_mat = np.array(data["M"])

from wmr_pure_pursuit import PurePursuitControl
controller = PurePursuitControl(kp=1,Lfc=10)

def execute(change):
    global robot, frames
    print("\rFrames", frames, end="")
    frames += 1
    
    
    img = cv2.resize(change["new"],(640,360)) #row col
    h, w = img.shape[:2]
    # undistort
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # Detect Red line
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_red2 = np.array([180,255,255])
    hsv_red1 = np.array([0,125,70])
    red = cv2.inRange(hsv,hsv_red1,hsv_red2)

    # Find target point
    j=0
    k=0
    for i in range(640):
        if red[270,i] >0:
            j = j+i
            k = k+1
    if k !=0 :    
        center = j/k
    else:
        center = 0
 
    # Transform to IPM
    #  - red line center 
    ipm_ = np.dot(proj_mat,np.array([center,270,1]))
    ipm_ = ipm_/ ipm_[2]
    #   - crnter of image
    ipm_ref = np.dot(proj_mat,np.array([320,270,1]))
    ipm_ref = ipm_ref/ ipm_ref[2]
    #   -the bottom of image
    ipm_cam = np.dot(proj_mat,np.array([640,270,1]))
    ipm_cam =  ipm_cam/ ipm_cam[2] 
    # Visualize
    _map = cv2.bitwise_and(img, img, mask=red)
    cv2.circle(_map, (int(center),270), 6, (255,255,0))
    cv2.circle(_map, (320,270), 6, (255,0,0))

    warped = cv2.warpPerspective(img,proj_mat, (w,h))
    cv2.circle(warped, (int(ipm_[0]),int(ipm_[1])), 6, (255,255,0))
    cv2.circle(warped, (int(ipm_ref[0]),int(ipm_ref[1])), 6, (255,0,0))
    cv2.putText(warped, str((int(ipm_[0]),int(ipm_[1]))), (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)
    cv2.putText(warped, str((int(ipm_ref[0]),int(ipm_ref[1]))), (320, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(warped, str((int(ipm_cam[0]),int(ipm_cam[1]))), (320, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("warped", warped)
    cv2.imshow("camera", _map)
    ## Pure Pursuit Control
    next_w = controller.feedback(((ipm_ref[0]- ipm_[0])*0.001,0.191) ,v )
    if center !=0:
        value_l = v+car_width*next_w/2
        value_r = v-car_width*next_w/2
    else:
        value_l = 0
        value_r = 0
    robot.set_motor(value_l, value_r)
    ## Camera acce
    img = cv2.resize(change["new"],(640,360))
    cv2.imshow("camera", img)
    ret,objp,corners = detect_chessboard(img)
    if ret !=0:
        R, T = solve_dis(objp,corners,Matrix,k)
        print(R)
        print(T)
        print(math.sqrt(T[0]**2+T[1]**2+T[2]**2))

def detect_chessboard(image):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    objp = np.zeros((3*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:3].T.reshape(-1,2)*20

    img = image
    h,w,c = img.shape
    bytesPerLine = 3*w
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,3),None) #
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


robot = Robot()
camera = Camera()
camera.observe(execute)
