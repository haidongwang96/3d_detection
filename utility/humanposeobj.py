"""
人体姿态检测与跟踪模块。
提供用于管理边界框、关键点和跟踪的专用类。
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

import utility as su
import camera


class BoundingBoxManager:
    """
    边界框管理类，处理检测框的创建、转换和匹配。
    
    Attributes:
        bbox (Tuple[float, float, float, float]): 边界框坐标 (x1, y1, x2, y2)
        confidence (float): 检测置信度
        class_id (float): 类别ID
        region (su.LabeledBoundingBox): 边界框对象
    """
    def __init__(self, bbox_data: np.ndarray):
        """
        初始化边界框管理器。
        
        Args:
            bbox_data (np.ndarray): 边界框数据，格式为[x1, y1, x2, y2, confidence, class_id]
        """
        self.bbox = tuple(bbox_data[:4])
        self.confidence = float(bbox_data[4])
        self.class_id = float(bbox_data[5]) if len(bbox_data) > 5 else 0.0
        self.region = self._create_bounding_box()

    
    def _create_bounding_box(self, expand: bool = False) -> su.LabeledBoundingBox:
        """
        创建标记的边界框对象。
        
        Args:
            expand (bool): 是否扩展边界框
            
        Returns:
            su.LabeledBoundingBox: 边界框对象
        """
        lbb = su.LabeledBoundingBox(bbox=self.bbox, label="person", score=self.confidence)
        if expand:
            lbb = lbb.expand(ratio_h=0.05, ratio_w=0.05)
        return lbb
    
    @property
    def tracker_format(self) -> Tuple[List[float], float, str]:
        """
        转换为跟踪器所需的格式。
        
        Returns:
            Tuple: ([left, top, width, height], confidence, label)
        """
        x, y = self.region.top_left
        w, h = self.region.width, self.region.height
        return [x, y, w, h], self.region.score, self.region.label
    
    def get_bottom_center(self) -> List[int]:
        """
        获取边界框底部中心点的2D坐标。
        
        Returns:
            List[int]: [x, y] 坐标
        """
        x_min, y_min, x_max, y_max = self.bbox
        mid_x = (x_min + x_max) / 2
        return [int(mid_x), int(y_max)]
    
    def project_to_ground(self, intrinsic: np.ndarray, rotation: np.ndarray, 
                          translation: np.ndarray) -> np.ndarray:
        """
        将边界框底部中心点投影到地面(z=0平面)。
        
        Args:
            intrinsic (np.ndarray): 相机内参矩阵
            rotation (np.ndarray): 相机旋转矩阵
            translation (np.ndarray): 相机平移向量
            
        Returns:
            np.ndarray: 地面上的3D位置  
        """
        p2d = self.get_bottom_center()
        #p2d[1] = p2d[1] - 10  # 减去10，因为y轴向下
        p3d_z0 = camera.project_bbox_bottom_center_to_z0_plane(p2d, intrinsic, rotation, translation)
        return p3d_z0


class KeypointManager:
    """
    关键点管理类，处理姿态关键点的过滤和匹配。
    
    Attributes:
        keypoints (np.ndarray): 关键点数组，形状为(N, 3)，每行为(x, y, confidence)
        valid_mask (np.ndarray): 有效关键点掩码
    """
    def __init__(self, keypoints: Optional[np.ndarray] = None):
        """
        初始化关键点管理器。
        
        Args:
            keypoints (np.ndarray, optional): 关键点数据，形状为(N, 3)
        """
        self.keypoints = keypoints
        self.valid_mask = np.ones(keypoints.shape[0], dtype=bool) if keypoints is not None else None
    
    def filter_by_confidence(self, threshold: float = 0.5) -> None:
        """
        根据置信度过滤关键点。
        
        Args:
            threshold (float): 置信度阈值
        """
        if self.keypoints is not None:
            data = self.keypoints.copy()
            data[:, 2] = (data[:, 2] >= threshold).astype(int)
            self.keypoints = data
            self.valid_mask = data[:, 2] > 0
    
    def get_valid_keypoints(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取有效关键点坐标。
        
        Args:
            indices (np.ndarray, optional): 指定的关键点索引，如果为None则返回所有有效关键点
            
        Returns:
            np.ndarray: 关键点坐标，形状为(N, 2)
        """
        if self.keypoints is None:
            return np.array([])
        
        if indices is None:
            valid_indices = np.where(self.valid_mask)[0]
        else:
            valid_indices = indices
        
        return self.keypoints[valid_indices, :2]
    
    @staticmethod
    def find_common_valid_indices(keypoint_managers: List['KeypointManager']) -> np.ndarray:
        """
        找到多个关键点管理器中共同有效的关键点索引。
        
        Args:
            keypoint_managers (List[KeypointManager]): 关键点管理器列表
            
        Returns:
            np.ndarray: 共同有效的关键点索引
        """
        if not keypoint_managers or any(km.keypoints is None for km in keypoint_managers):
            return np.array([])
        
        masks = [km.keypoints[:, 2] == 1 for km in keypoint_managers]
        masks = np.array(masks)
        common_mask = np.all(masks, axis=0)
        return np.where(common_mask)[0]


class TrackManager:
    """
    跟踪管理类，处理目标跟踪和轨迹管理。
    
    Attributes:
        track: 跟踪对象
        track_id (int): 跟踪ID
        bbox_manager (BoundingBoxManager): 边界框管理器
        keypoint_manager (KeypointManager): 关键点管理器
        world_position (np.ndarray): 世界坐标系中的位置
    """
    def __init__(self, track: Any, bbox_manager: BoundingBoxManager, 
                 keypoint_manager: Optional[KeypointManager] = None):
        """
        初始化跟踪管理器。
        
        Args:
            track: 跟踪对象
            bbox_manager (BoundingBoxManager): 边界框管理器
            keypoint_manager (KeypointManager, optional): 关键点管理器
        """
        self.track = track
        self.track_id = track.track_id
        self.bbox_manager = bbox_manager
        self.keypoint_manager = keypoint_manager
        self.world_position = None

    def get_track_box_center(self) -> List[int]:
        """
        获取跟踪框中心点。
        
        Returns:
            List[int]: [x, y] 坐标
        """
        x_min, y_min, x_max, y_max = self.track.to_tlbr()
        mid_x = (x_min + x_max) / 2 
        mid_y = (y_min + y_max) / 2
        return [int(mid_x), int(mid_y)]
    

    def get_track_box_bottom_center(self) -> List[int]:
        """
        获取跟踪框底部中心点。
        
        Returns:
            List[int]: [x, y] 坐标
        """
        x_min, y_min, x_max, y_max = self.track.to_tlbr()
        mid_x = (x_min + x_max) / 2
        return [int(mid_x), int(y_max)]
    
    def update_world_position(self, intrinsic: np.ndarray, rotation: np.ndarray, 
                             translation: np.ndarray) -> np.ndarray:
        """
        更新目标在世界坐标系中的位置。
        
        Args:
            intrinsic (np.ndarray): 相机内参矩阵
            rotation (np.ndarray): 相机旋转矩阵
            translation (np.ndarray): 相机平移向量
            
        Returns:
            np.ndarray: 更新后的世界坐标
        """
        self.world_position = self.bbox_manager.project_to_ground(intrinsic, rotation, translation)
        return self.world_position


class HumanPoseObject:
    """
    人体姿态对象，整合边界框和关键点信息。
    
    Attributes:
        bbox_manager (BoundingBoxManager): 边界框管理器
        keypoint_manager (KeypointManager): 关键点管理器
    """
    def __init__(self, boxes: np.ndarray, keypoints: Optional[np.ndarray] = None):
        """
        初始化人体姿态对象。
        
        Args:
            boxes (np.ndarray): 边界框数据
            keypoints (np.ndarray, optional): 关键点数据
        """
        self.bbox_manager = BoundingBoxManager(boxes)
        self.keypoint_manager = KeypointManager(keypoints)
        
        # 为了向后兼容
        self.box = self.bbox_manager.bbox
        self.region = self.bbox_manager.region
        self.keypoints = keypoints
    
    @property
    def region2trackerformat(self):
        """向后兼容的接口"""
        return self.bbox_manager.tracker_format
    
    def keypoints_reform_by_thresh(self, threshold: float = 0.5):
        """向后兼容的接口"""
        self.keypoint_manager.filter_by_confidence(threshold)
        self.keypoints = self.keypoint_manager.keypoints
    
    def valid_keypoints_extract(self, indices: np.ndarray):
        """向后兼容的接口"""
        return self.keypoint_manager.get_valid_keypoints(indices)


class TrackerManager:
    """
    跟踪管理类，处理跟踪对象的创建和更新。
    """
    def __init__(self, tracker):
        """
        初始化跟踪管理器。
        
        Args:
            tracker: deepsort tracker
        """
        self.tracker = tracker
        self.detection_objects = []


    def update_tracker(self, tracker):
        """
        更新跟踪器。
        """
        self.tracker = tracker

    @property
    def get_tracks(self):   
        return self.tracker.tracks
    


class TrackHPObj():
    """
    跟踪的人体姿态对象，结合跟踪信息和姿态检测结果。
    向后兼容的类。
    """
    def __init__(self, track, obj: HumanPoseObject):
        """
        初始化跟踪的人体姿态对象。
        
        Args:
            track: 跟踪对象
            obj (HumanPoseObject): 人体姿态对象
        """
        self.track = track
        self.obj = obj
        self.track_world_coord = None
        self.det_world_coord = None
        
        # 创建新的管理器
        self.track_manager = TrackManager(
            track, 
            obj.bbox_manager, 
            obj.keypoint_manager
        )

        self.feature = None
        self.p3d_z0_det = None
        self.p3d_z0_track = None

    @property
    def get_track_box_bottom_center(self):
        """向后兼容的接口"""
        return self.track_manager.get_track_box_bottom_center()
    
    @property
    def get_track_box_center(self):
        """向后兼容的接口"""
        return self.track_manager.get_track_box_center()
    
    @property
    def get_obj_box_bottom_center(self):
        """向后兼容的接口"""
        return self.obj.bbox_manager.get_bottom_center()
    
    def obj_floor_coordinate(self, k, R, t):
        """向后兼容的接口"""
        self.det_world_coord = self.obj.bbox_manager.project_to_ground(k, R, t)
        return self.det_world_coord
    
    def update_track_world_coord(self, k, R, t):
        """向后兼容的接口"""
        self.track_world_coord = self.track_manager.update_world_position(k, R, t)
        return self.track_world_coord


class HumanPoseObject_3d:
    """
    多视角人体姿态对象，用于融合多个视角的姿态信息。
    向后兼容的类。
    """
    def __init__(self, objs: List[HumanPoseObject]):
        """
        初始化多视角人体姿态对象。
        
        Args:
            objs (List[HumanPoseObject]): 多视角姿态对象列表
        """
        self.objs = objs
        self.common_indices = None
        self.matched_kypts = None
        self.p3d_z0_det = []

        # 创建关键点管理器列表
        self.keypoint_managers = [obj.keypoint_manager for obj in objs]
    
    def valid_matched_kypts(self):
        """向后兼容的接口"""
        self.multi_view_keypoints_reform_by_thresh()
        self.common_indices = KeypointManager.find_common_valid_indices(self.keypoint_managers)
        
        temp = []
        for obj in self.objs:
            kypt = obj.valid_keypoints_extract(self.common_indices)
            temp.append(kypt)
        
        self.matched_kypts = np.array(temp)
        return self.matched_kypts
    
    def multi_view_keypoints_reform_by_thresh(self, threshold=0.5):
        """向后兼容的接口"""
        for obj in self.objs:
            obj.keypoints_reform_by_thresh(threshold)
    
    def extract_valid_kypts(self):
        """向后兼容的接口"""
        masks = []
        for obj in self.objs:
            mask_i = obj.keypoints[:, 2] == 1
            masks.append(mask_i)
        
        masks = np.array(masks)
        common_mask = np.all(masks, axis=0)
        common_indices = np.where(common_mask)[0]
        return common_indices


if __name__ == '__main__':
    np.random.seed(42)  # 确保示例可重复
    example_bbox1 = np.random.rand(6, )
    example_kypt1 = np.random.rand(17, 3)  # 第三列是0到1的随机数

    example_bbox2 = np.random.rand(6,)
    example_kypt2 = np.random.rand(17,3)

    obj1 = HumanPoseObject(example_bbox1, example_kypt1)
    obj2 = HumanPoseObject(example_bbox2, example_kypt2)

    obj_3d = HumanPoseObject_3d([obj1, obj2])
    kypts = obj_3d.valid_matched_kypts()
    print(obj_3d.common_indices)
    print(kypts.shape)










