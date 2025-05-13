import os
import sys
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径
import camera

"""
1.Aruco_0 为同一时刻两个相机拍摄的同一张的aruco码， 两个相机要求清晰可见
2.Chessboard 分别对两个摄像头进行拍摄不同角度的棋盘格图像，不要求同时刻，同一张，只要求清晰可见
3.Landmark 拍摄一张即可, landmark 进行标注。 要求，only四点，按照顺序点出四个角点，点在两幅图对应位置
"""

def multi_camera_calibrate_intrinsics(sample_folder,save_dir, cam_ids=[0,1]):
    """
    依次校准同一次拍摄的相机 0&1 的内参
    :param sample_folder: 输入两个相机同时拍摄的文件夹
    """

    for cam_id in cam_ids:
        single_camera_calibrate_intrinsics(sample_folder, save_dir, cam_id)
        single_camera_calibrate_intrinsics(sample_folder, save_dir, cam_id)

def single_camera_calibrate_intrinsics(sample_folder, save_dir,cam_id):
    """
    校准单个相机(cam_id)的内参
    """
    #camera.single_camera_calibrate_intrinsic_parameters(sample_folder, cam_id) # for visualization
    mtx, dist = camera.single_camera_calibrate_intrinsic_redo_with_rmse(sample_folder,cam_id)
    camera.save_camera_intrinsics(mtx, dist, cam_id, save_dir,option="json")


def multi_camera_calibrate_extrinsics(image_folder_path, intrinsic_dir, cam_ids= [0, 1]):
    camera.single_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_ids[0])
    camera.single_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_ids[1])


def multi_camera_landmark_annotation(landmark_dir, cam_ids=[0, 1]):

    """
    only四点
    按照顺序点出四个角点
    点在两幅图对应位置
    """

    for cam_id in cam_ids:
        print(cam_id)
        camera.single_landmark_annotation(landmark_dir, cam_id)

if __name__ == '__main__':


    cam_ids = [0, 1]
    record_root_folder = "video_test_0"
    print(os.listdir(record_root_folder))
    intrinsic_folder = os.path.join(record_root_folder, "calib_info")

    # 棋盘格
    # 内参标定
    # chessboard_folder = os.path.join(record_root_folder, "chessboard_0")
    # multi_camera_calibrate_intrinsics(chessboard_folder, intrinsic_folder, cam_ids=cam_ids)

    # arUco码
    # 双相机外参同时标定
    # aruco_folder = os.path.join(record_root_folder, "aruco_0")
    # multi_camera_calibrate_extrinsics(aruco_folder, intrinsic_folder,cam_ids=cam_ids)

    # fence标定
    landmark_folder = os.path.join(record_root_folder, "landmark_0")
    print(landmark_folder)
    for cam_id in cam_ids:
        print(cam_id)
        camera.single_landmark_annotation(record_root_folder,landmark_folder, cam_id)
    #multi_camera_landmark_annotation(landmark_folder, cam_ids)



