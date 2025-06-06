import cv2
import glob
import os
import scipy
import numpy as np

import camera
import utility as su

def single_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_id):
    """
    使用单张aruco图像，对相机到此张aruco的外参坐标进行标定
    :return:
    """

    images_names = su.collect_images_by_index(image_folder_path, cam_id)
    assert len(images_names) == 1, ("外参为唯一图像")
    aurco_image = images_names[0]

    intrinsic_file_path = os.path.join(intrinsic_dir, f'camera_intrinsic_cam_{cam_id}.json')
    mtx, dist = camera.load_intrinsic_calibration_parameters(intrinsic_file_path)

    frame = cv2.imread(aurco_image)
    # 依次校准，分别得到相机的旋转矩阵，平移向量（aurco坐标系 -> 相机坐标系)
    rvec, tvec = camera.estimate_single_marker_r_t(frame, mtx)

    # 将aruco显示到frame上
    frame_marked = camera.estimate_pose(frame, mtx, dist)
    # cv2.imshow('frame', frame_marked)
    # cv2.waitKey(0)  # 0 means wait indefinitely
    # cv2.destroyAllWindows()

    extr = camera.Extrinsic(rvec, tvec)
    print(f"相机{cam_id}外参：")
    print(extr)
    # aurco坐标系 -> 相机坐标系
    save_dir = intrinsic_dir
    extr_path_aruco_c = os.path.join(save_dir, f"camera_extrinsic_arUco2cam_{cam_id}.json")
    extr.save(extr_path_aruco_c)
    image_path_aruco_c = os.path.join(save_dir, f"camera_extrinsic_show_arUco2cam_{cam_id}.jpg")
    cv2.imwrite(image_path_aruco_c,frame_marked)

def multi_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_ids= [0, 1]):
    single_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_ids[0])
    single_camera_calibrate_extrinsic(image_folder_path, intrinsic_dir, cam_ids[1])



def single_camera_calibrate_intrinsic_redo_with_rmse(image_folder_path, cam_id):
    # NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = su.collect_images_by_index(image_folder_path, cam_id)

    # read all frames
    images = [cv2.imread(imname, 1) for imname in images_names]
    print(f"{len(images)} images found.")

    # 初始化对象点和图像点列表
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.

    # 设置棋盘格的规格
    chessboard_size = (9, 6)  # 棋盘格内角点数目

    # 准备棋盘格对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)


    # 遍历所有的图像文件路径
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将当前图像转换为灰度图

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到棋盘格角点
        if ret:
            # 将棋盘格角点添加到imgpoints列表中
            imgpoints.append(corners)
            # 添加相应的3D对象点
            objpoints.append(objp)


    # 使用对象点和图像点进行相机标定，得到相机的内参矩阵、畸变系数以及旋转矩阵和平移矩阵
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 剔除重投影误差大于0.3的图像
    new_objpoints = []
    new_imgpoints = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        if error <= 0.3:
            new_objpoints.append(objpoints[i])
            new_imgpoints.append(imgpoints[i])

    # 使用剔除后的点重新进行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(new_objpoints, new_imgpoints, gray.shape[::-1], None, None)

    # 计算重投影误差
    mean_error = 0
    for i in range(len(new_objpoints)):
        imgpoints2, _ = cv2.projectPoints(new_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(new_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"相机{cam_id}内参：")
    print(mtx)
    print("剔除后畸变系数")
    print(dist)
    # 输出重投影误差
    print("剔除后重投影", mean_error / len(new_objpoints))

    return mtx, dist



# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def single_camera_calibrate_intrinsic_parameters(image_folder_path, cam_id):

    # NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = su.collect_images_by_index(image_folder_path, cam_id)

    # read all frames
    images = [cv2.imread(imname, 1) for imname in images_names]
    print(f"{len(images)} images found.")

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard. 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Calibration board needs to match inputs rows & columns #
    #  rows & columns # are the intersection dot of white and black squares
    chessboard_size = (6, 9, 0.022)
    rows = chessboard_size[0]
    columns = chessboard_size[1]
    world_scaling = chessboard_size[2]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv2 can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('img', frame)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    #print(imgpoints[:1])
    cv2.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist





# open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1, calibration_settings):
    # read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))
    print(c0_images_names)
    print(c1_images_names)

    # open images
    c0_images = [cv2.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv2.imread(imname, 1) for imname in c1_images_names]

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv2.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv2.imshow('img', frame0)

            cv2.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv2.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv2.imshow('img2', frame1)
            k = cv2.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    # print(objp[0])
    # print(imgpoints_left[0])
    # print(imgpoints_right[0])

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0,
                                                                  dist0,
                                                                  mtx1, dist1, (width, height), criteria=criteria,
                                                                  flags=stereocalibration_flags)

    print('rmse: ', ret)
    print('CM1: ', CM1)
    print('CM2: ', CM2)
    print('R: ', R)
    print('T: ', T)

    np.save('stereo.npy', {'Kl': CM1, 'Dl': dist0, 'Kr': CM2, 'Dr': dist1, 'R': R, 'T': T, 'E': E, 'F': F,
                           'img_size': (width, height), 'left_pts': imgpoints_left, 'right_pts': imgpoints_right})

    cv2.destroyAllWindows()
    return R, T


# save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, save_dir,  option="json"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save .json format
    if option == "json":
        out_filename_name = f"camera_intrinsic_cam_{camera_name}.json"
        save_path = os.path.join(save_dir, out_filename_name)
        save_dict = {"intrinsic": camera_matrix.tolist(), "distortion_coefs": distortion_coefs.tolist()}
        print(f"intrinsic saved:{save_path}")
        print(save_dict)
        su.write_json_file(save_dict, save_path)
    else:
        print("system only support json")


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix='', option="json"):
    assert option in ["dat", "json"], (f"option {option} not supported")

    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    # save .dat format
    if option == "dat":
        camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_extrinsics.dat')
        outf = open(camera0_rot_trans_filename, 'w')

        outf.write('R:\n')
        for l in R0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('T:\n')
        for l in T0:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.close()

        # R1 and T1 are just stereo calibration returned values
        camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_extrinsics.dat')
        outf = open(camera1_rot_trans_filename, 'w')

        outf.write('R:\n')
        for l in R1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')

        outf.write('T:\n')
        for l in T1:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.close()

    # save .json format
    if option == "json":
        camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_extrinsics.json')
        out_c0_extrin = {}
        out_c0_extrin["R"] = R0.tolist()
        out_c0_extrin["T"] = T0.tolist()
        su.write_json_file(out_c0_extrin, camera0_rot_trans_filename)

        # R1 and T1 are just stereo calibration returned values
        camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_extrinsics.json')
        out_c1_extrin = {}
        out_c1_extrin["R"] = R1.tolist()
        out_c1_extrin["T"] = T1.tolist()
        su.write_json_file(out_c1_extrin, camera1_rot_trans_filename)


def load_intrinsic_calibration_parameters(path):
    d = su.read_json_file(path)
    cmtx = np.array(d['intrinsic'])
    dist = np.array(d['distortion_coefs'])

    return cmtx, dist


def load_extrinsic_calibration_parameters(path):
    d = su.read_json_file(path)
    R = np.array(d['R'])
    T = np.array(d['T'])

    return R, T


# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


def get_extrinsic_3_4(R, t):
    E = make_homogeneous_rep_matrix(R, t)
    return E[:3, :]


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


def get_projection_matrix2(cmtx, E):
    P = cmtx @ E[:3, :]
    return P


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = scipy.linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]


def project_3D_points_to_camera(draw_axes_points, P):
    pixel_points_camera = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])

        # project to camera0
        uv = P @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera.append(uv)

    return pixel_points_camera





if __name__ == '__main__':
    pass
