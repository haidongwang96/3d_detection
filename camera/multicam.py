import cv2
import os
import time
import yaml
import numpy as np
from typing import List, Dict, Optional, Tuple

import camera
import utility as su


class VirtualFences:

    # 拍摄时处于同一时刻，放置在同一文件夹

    def __init__(self, config):
        self.config = config
        self.landmark_folder = os.path.join(config['root'], config['proj_root'], config['landmark'])
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




class CameraManager:
    """
    相机管理类，用于管理多相机系统的参数和坐标变换
    
    功能：
    1. 管理相机内外参
    2. 缓存投影矩阵等计算结果
    3. 提供坐标变换接口
    4. 支持多相机系统
    """
    def __init__(self, config):
        """
        初始化相机管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = config
        self.cam_ids = self.config["cam_ids"]
        self.proj_root = self.config["proj_root"]
        
        # 初始化相机参数存储
        self._init_camera_params()
        
        # 缓存计算结果
        self._cache = {}
        
    def _init_camera_params(self):
        """初始化所有相机的参数"""

        calib_folder = os.path.join(self.config['root'], self.config['proj_root'], 'calib_info')
        
        self.cameras = {}
        self.intrinsics = {}
        self.extrinsics = {}
        self.projection_matrices = {}
        
        for cam_id in self.cam_ids:
            # 加载内参和外参
            intr_path = f"{calib_folder}/camera_intrinsic_cam_{cam_id}.json"
            extr_path = f"{calib_folder}/camera_extrinsic_landmark_{cam_id}.json"
            
            # 创建相机对象
            cam = camera.SingleCamera(intr_path=intr_path, extr_path=extr_path)
            
            # 存储参数
            self.cameras[cam_id] = cam
            self.intrinsics[cam_id] = cam.intr.get_cam_mtx()
            self.extrinsics[cam_id] = {
                'R': cam.extr.R(),
                't': cam.extr.t(),
                'r_vec': cam.extr.r_vec(),
                'pose_mat': cam.extr.pose_mat
            }
            self.projection_matrices[cam_id] = cam.projection_matrix
    
    def get_camera(self, cam_id: int) -> SingleCamera:
        """获取指定ID的相机对象"""
        return self.cameras[cam_id]
    
    def get_projection_matrix(self, cam_id: int) -> np.ndarray:
        """获取指定相机的投影矩阵"""
        return self.projection_matrices[cam_id]
    
    def get_camera_params(self, cam_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取指定相机的内参和外参
        
        Returns:
            tuple: (内参矩阵K, 旋转矩阵R, 平移向量t)
        """
        return (
            self.intrinsics[cam_id],
            self.extrinsics[cam_id]['R'],
            self.extrinsics[cam_id]['t']
        )
    
    def world_to_image(self, cam_id: int, points_3d: np.ndarray) -> np.ndarray:
        """
        将世界坐标系下的3D点投影到指定相机的图像平面
        
        Args:
            cam_id: 相机ID
            points_3d: 形状为(N,3)的3D点坐标数组
            
        Returns:
            np.ndarray: 形状为(N,2)的图像坐标数组
        """
        cache_key = f'world_to_image_{cam_id}'
        if cache_key not in self._cache:
            K = self.intrinsics[cam_id]
            R = self.extrinsics[cam_id]['R']
            t = self.extrinsics[cam_id]['t']
            self._cache[cache_key] = {'K': K, 'R': R, 't': t}
        
        params = self._cache[cache_key]
        points_2d = cv2.projectPoints(
            points_3d,
            params['R'],
            params['t'],
            params['K'],
            None
        )[0].reshape(-1, 2)
        
        return points_2d
    
    def image_to_world_ground(self, cam_id: int, points_2d: np.ndarray) -> np.ndarray:
        """
        todo: not tested
        将图像平面上的点反投影到世界坐标系的地平面(Z=0)上
        
        Args:
            cam_id: 相机ID
            points_2d: 形状为(N,2)的图像坐标数组
            
        Returns:
            np.ndarray: 形状为(N,3)的3D点坐标数组(Z=0)
        """
        K = self.intrinsics[cam_id]
        R = self.extrinsics[cam_id]['R']
        t = self.extrinsics[cam_id]['t']
        
        # 计算相机坐标系下的射线方向
        points_2d_homo = np.hstack([points_2d, np.ones((len(points_2d), 1))])
        rays = np.linalg.inv(K) @ points_2d_homo.T
        
        # 转换到世界坐标系
        rays = (R.T @ rays).T
        camera_center = -R.T @ t
        
        # 计算射线与地平面(Z=0)的交点
        scale = -camera_center[2] / rays[:, 2]
        points_3d = camera_center + scale.reshape(-1, 1) * rays
        
        return points_3d
    
    def triangulate_points(self, points_2d_1: np.ndarray, points_2d_2: np.ndarray,
                          cam_id_1: int, cam_id_2: int) -> np.ndarray:
        """
        三角测量两个相机视角下的对应点
        
        Args:
            points_2d_1: 第一个相机视角下的2D点坐标 (N,2)
            points_2d_2: 第二个相机视角下的2D点坐标 (N,2)
            cam_id_1: 第一个相机ID
            cam_id_2: 第二个相机ID
            
        Returns:
            np.ndarray: 三角测量得到的3D点坐标 (N,3)
        """
        P1 = self.projection_matrices[cam_id_1]
        P2 = self.projection_matrices[cam_id_2]
        
        points_4d = cv2.triangulatePoints(P1, P2, points_2d_1.T, points_2d_2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d
    
    
    

class MultiCameraCapture:
    """
    多相机视频流管理类
    
    专注于：
    1. 视频文件/相机设备的读取管理
    2. 帧同步和读取
    3. 视频流参数设置
    """
    def __init__(self, config: dict):
        """
        初始化多相机视频流管理器
        
        Args:
            config: 配置字典，包含：
                - cam_ids: 相机ID列表
                - proj_root: 项目根目录
                - cam_record: 相机录制参数（宽度、高度、帧率等）
        """
        self.config = config
        self.cam_ids = config['cam_ids']
        self.proj_root = config['proj_root']
        
        # 视频流参数
        self.stream_params = {
            'width': config['cam_record']['WIDTH'],
            'height': config['cam_record']['HEIGHT'],
            'fps': config['cam_record']['FPS']
        }
        
        # 视频捕获器
        self.captures = None
        
        # 帧计数和时间统计
        self._frame_count = 0
        self._start_time = time.time()
        
    def video_feeds(self, video_feed_ts: str, video_folder: Optional[str] = None) -> None:
        """
        设置视频源
        
        Args:
            video_feed_ts: 视频时间戳
            video_folder: 可选的视频文件夹名
        """
        feeds = []
        for cam_id in self.cam_ids:
            # 构建视频文件路径
            video_path = f"{video_folder}/{video_feed_ts}_{cam_id}.mp4"

            # 创建视频捕获器
            cap = self._create_video_capture(video_path)
            feeds.append(cap)
            print(f"加载视频: {video_path}")
            
        self.captures = feeds
        self._verify_captures()
    
    def camera_feeds(self, device_ids: List[int]) -> None:
        """
        设置实时相机输入
        
        Args:
            device_ids: 相机设备ID列表
        """
        assert len(device_ids) == len(self.cam_ids), "设备数量与配置的相机数量不匹配"
        
        feeds = []
        for dev_id in device_ids:
            cap = self._create_video_capture(dev_id)
            feeds.append(cap)
            print(f"连接相机设备: {dev_id}")
            
        self.captures = feeds
        self._verify_captures()
    
    def _create_video_capture(self, source) -> cv2.VideoCapture:
        """
        创建并配置视频捕获器
        
        Args:
            source: 视频源（可以是文件路径或设备ID）
            
        Returns:
            cv2.VideoCapture: 配置好的视频捕获器
        """
        cap = cv2.VideoCapture(source)
        
        # 设置视频流参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_params['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_params['height'])
        cap.set(cv2.CAP_PROP_FPS, self.stream_params['fps'])
        
        return cap
    
    def _verify_captures(self) -> None:
        """验证所有视频捕获器是否正常打开"""
        if not self.captures:
            raise ValueError("未设置视频源")
            
        for i, cap in enumerate(self.captures):
            if not cap.isOpened():
                raise ValueError(f"无法打开相机/视频 {self.cam_ids[i]}")
            print(f"成功初始化摄像头 {self.cam_ids[i]}")
    
    def read_frames(self) -> List[np.ndarray]:
        """
        同步读取所有相机的当前帧
        
        Returns:
            List[np.ndarray]: 所有相机的当前帧图像列表
        """
        if not self.captures:
            raise ValueError("未初始化视频源，请先调用 video_feeds 或 camera_feeds")
        
        frames = []
        for cap in self.captures:
            ret, frame = cap.read()
            if not ret:
                return None
            frames.append(frame)
        
        self._frame_count += 1
        return frames
    
    def get_fps(self) -> float:
        """
        计算当前的平均帧率
        
        Returns:
            float: 平均帧率
        """
        elapsed_time = time.time() - self._start_time
        return self._frame_count / elapsed_time if elapsed_time > 0 else 0
    
    def release(self) -> None:
        """释放所有视频资源"""
        if self.captures:
            for cap in self.captures:
                cap.release()
            self.captures = None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源释放"""
        self.release()



# 示例用法
if __name__ == "__main__":
    pass