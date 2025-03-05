import numpy as np
import camera

def get_3d_landmark(landmark0, landmark1, P0, P1):
    landmark_3d = camera.triangulation_cv2(P0, P1, landmark0, landmark1)
    return landmark_3d

