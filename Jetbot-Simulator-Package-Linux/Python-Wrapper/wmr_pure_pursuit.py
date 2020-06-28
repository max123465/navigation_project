import numpy as np 
import math
class PurePursuitControl:
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    def set_path(self, path):
        self.path = path.copy()

    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    def _search_target(self, pos,Ld,ind_near):
        min_dist = 99999999
        min_id = -1
	
        for i in range(ind_near,self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if (dist -Ld) < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist
    def feedback(self, target,v):
        # Check Path
        if target is None:
            print("No target !!")
            return None, None
        
        # Extract State 

        target_x = target[0]	
        target_y = target[1]
        ### hint: (you first need to find the nearest point and then find the point(target) backward, this will make your model won't go back)
        ### hint: (if you can not find a point(target) on the path which distance between the path and model is as same as the Ld, you need to find a similar one)
        # third, you need to calculate alpha
        alpha = math.atan(-target_x/target_y)  # rotate 90
        Ld = np.sqrt(target_x**2 + target_y**2)
        R = Ld/2/np.sin(alpha)
        # now, you can calculate the w
        next_w = v/R

        # The next_w is Pure Pursuit Control's output
        # The target is the point on the path which you find
        #####################################################################
        return next_w

if __name__ == "__main__":
    import cv2
    import path_generator
    import sys
    sys.path.append("../")
    from wmr_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    # Initialize Car
    car = KinematicModel()
    car.init_state((50,300,0))
    controller = PurePursuitControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str()+"\t")

        # ================= Control Algorithm ================= 
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 40 if end_dist > 265 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v}
        next_delta = controller.feedback(target)
        car.control(next_a,next_delta)
        # =====================================================
        
        # Update & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Pure-Pursuit Control Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            _init_state(car)
        if k == 27:
            print()
            break
