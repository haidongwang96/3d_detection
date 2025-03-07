import preprocessing_tools
import os
import camera

intr = False
landm = True

"""
以tri_test为例
cam# 为手动标记 # 为第几个实物摄像头
folder内的id 为cv2 readin 的usb id
"""
record_root_folder = "../data/record/tri_test"
cam_param_folder = os.path.join(record_root_folder, "calib_info")

if intr:
    # chessboard for intrinsic
    chessboard_folder = os.path.join(record_root_folder, "cam3")
    preprocessing_tools.single_camera_calibrate_intrinsics(chessboard_folder, cam_param_folder, 5)

if landm:
    # landmark/外参标定
    # 必须提前标定过内参
    landmark_folder = os.path.join(record_root_folder, "landmark_2")
    camera.single_landmark_annotation(record_root_folder, landmark_folder, 7)


