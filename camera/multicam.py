import cv2
import os
import time
import yaml
import numpy as np

import camera
import utility as su


class VirtualFences:

    # 拍摄时处于同一时刻，放置在同一文件夹

    def __init__(self, config):
        self.config = config
        self.landmark_folder = f"../data/record/{self.config['proj_root']}/landmark_0"
        self.landmarks = self.load_virtual_fences(self.landmark_folder, config["cam_ids"])

    def load_virtual_fences(self, landmark_dir, cam_ids):
        landmarks = []
        for cam_id in cam_ids:
            landmark_file = su.collect_file_by_index_prefix(landmark_dir, cam_id, prefix="txt")
            landmark_file = landmark_file[0]
            landmark = su.read_list_file(landmark_file, " ")
            landmark = np.array(landmark, dtype=int)
            landmarks.append(landmark)
        return landmarks


class SingleCamera:

    def __init__(self, intr_path=None, extr_path=None, intr:camera.Intrinsics=None, extr:camera.Extrinsic=None):
        assert intr_path or intr is not None
        if intr_path is not None:
            self.intr = camera.intr_load(intr_path)
        else:
            self.intr = intr

        assert extr_path or extr is not None
        if extr_path is not None:
            # aruco to camera
            self.extr = camera.extr_load(extr_path)
        else:
            self.extr = extr
        # 3*3 @ 3*4
        self.projection_matrix = self.intr.get_cam_mtx() @ self.extr.pose_mat[:3, :]
        #self.projection_matrix = self.intr.get_cam_mtx() @ self.extr.inverse_transformation().pose_mat[:3, :]

    def identity_extr(self):
        # GET indentity extr
        return SingleCamera(intr=self.intr, extr=camera.get_self_transformation_extrinsic())

    def extr_to_other_cam(self, cam):
        # this(self）camera cam transform to other cam
        # cam1: this cam
        # cam2: input cam
        transformed_extr = camera.cam1_to_cam2_transformation(self.extr, cam.extr)
        return SingleCamera(intr=self.intr, extr=transformed_extr)





class MultiCameraCapture:

    def __init__(self, config):
        """
        初始化多个摄像头
        :param camera_indices: 摄像头索引列表，例如 [0, 1] 表示使用第一个和第二个摄像头
        """
        self.config = config
        self.cam_ids = self.config['cam_ids']
        self.proj_root = self.config['proj_root']
        self.calibration_folder = f"../data/record/{self.proj_root}/calib_info"
        self.Cameras = []
        for cam_id in self.cam_ids:
            intr_path = os.path.join(self.calibration_folder, f"camera_intrinsic_cam_{cam_id}.json")
            # solvePnP 返回的旋转是从世界坐标系到相机坐标系的旋转。
            extr_path = os.path.join(self.calibration_folder, f"camera_extrinsic_landmark_{cam_id}.json")
            self.Cameras.append(SingleCamera(intr_path=intr_path,extr_path=extr_path))

        self.captures = None

        # 初始化 FPS 计算相关变量
        self.frame_count = 0
        self.start_time = time.time()


    def video_feeds(self, video_feed_ts, video_folder=None):
        # 根据录制的timestamp
        feeds = []
        for id in self.cam_ids:
            if video_folder is  not None:
                video_i = f"../data/record/{self.proj_root}/{video_folder}/{video_feed_ts}_{id}.mp4"
            else:
                video_i = f"../data/record/{self.proj_root}/{video_feed_ts}_{id}.mp4"
            feeds.append(self.get_cv2_capture(video_i))
            print(video_i)
        self.captures = feeds
        # 检查摄像头是否成功打开
        for i, capture in enumerate(self.captures):
            if not capture.isOpened():
                raise ValueError(f"无法打开摄像头 {self.cam_ids[i]}")
            print(f" 成功读取==摄像头{self.cam_ids[i]}==")

    def read_frames(self):
        """
        从所有摄像头读取帧
        :return: 返回一个包含所有摄像头帧的列表
        """
        frames = []
        for capture in self.captures:
            ret, frame = capture.read()
            if not ret:
                raise ValueError("无法从摄像头读取帧")
            frames.append(frame)

        # 更新帧计数
        self.frame_count += 1
        return frames


    def get_cv2_capture(self, cam_path):
        """
        此函数的目的是同一所有opencv的视频流设定
        Note: 采用不同分辨率的相机，内参不同
        """
        cap = cv2.VideoCapture(cam_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['cam_record']["WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['cam_record']["HEIGHT"])
        cap.set(cv2.CAP_PROP_FPS, self.config['cam_record']["FPS"])  # 设置帧率为30fps

        return cap

    def release(self):
        """
        释放所有摄像头资源
        """
        for capture in self.captures:
            capture.release()

    # def __del__(self):
    #     """
    #     析构函数，确保摄像头资源被释放
    #     """
    #     self.release()


class MultiCameraCapture_abondon:
    def __init__(self, camera_inputs):
        """
        初始化多个摄像头
        :param camera_indices: 摄像头索引列表，例如 [0, 1] 表示使用第一个和第二个摄像头
        """
        self.config = su.read_yaml_file('../camera/config.yaml')

        self.camera_inputs = camera_inputs
        self.captures = [self.get_cv2_capture(vi) for vi in camera_inputs]
        self.camera_indices = [i for i, _ in enumerate(self.camera_inputs)]

        # 不同摄像头的内外惨数
        self.intrs = [None for _ in self.camera_indices]
        self.extrs = [None for _ in self.camera_indices]
        self.projection_matrix = [None for _ in self.camera_indices]

        # 检查摄像头是否成功打开
        for i, capture in enumerate(self.captures):
            if not capture.isOpened():
                raise ValueError(f"无法打开摄像头 {self.camera_inputs[i]}")

        # 初始化 FPS 计算相关变量
        self.frame_count = 0
        self.start_time = time.time()

    def get_cv2_capture(self, cam_path):
        """
        此函数的目的是同一所有opencv的视频流设定
        Note: 采用不同分辨率的相机，内参不同
        """
        cap = cv2.VideoCapture(cam_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['cam_record']["WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['cam_record']["HEIGHT"])
        cap.set(cv2.CAP_PROP_FPS, self.config['cam_record']["FPS"])  # 设置帧率为30fps

        return cap

    def read_frames(self):
        """
        从所有摄像头读取帧
        :return: 返回一个包含所有摄像头帧的列表
        """
        frames = []
        for capture in self.captures:
            ret, frame = capture.read()
            if not ret:
                raise ValueError("无法从摄像头读取帧")
            frames.append(frame)

        # 更新帧计数
        self.frame_count += 1
        return frames


    def get_fps(self):
        """
        计算并返回当前的平均 FPS
        :return: 当前的平均 FPS
        """
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps

    def set_intrinsics(self, camera_index, intrinsic_matrix):
        """
        设置指定相机的内参
        :param camera_index: 相机索引[int]
        :param intrinsic_matrix: 内参矩阵
        """
        if camera_index in self.camera_indices:
            self.intrs[camera_index:int] = camera.Intrinsics(intrinsic_matrix)
        else:
            print(f"相机索引 {camera_index} 不存在")

    def set_extrinsics(self, camera_index, rotation_matrix, translation_vector):
        """
        设置指定相机的外参
        :param camera_index: 相机索引
        :param rotation_matrix: 旋转矩阵
        :param translation_vector: 平移向量
        """
        if camera_index in self.camera_indices:
            self.extrs[camera_index:int] = camera.Extrinsic(rotation_matrix, translation_vector)
        else:
            print(f"相机索引 {camera_index} 不存在")


    def release(self):
        """
        释放所有摄像头资源
        """
        for capture in self.captures:
            capture.release()

    # def __del__(self):
    #     """
    #     析构函数，确保摄像头资源被释放
    #     """
    #     self.release()

# 示例用法
if __name__ == "__main__":
    pass