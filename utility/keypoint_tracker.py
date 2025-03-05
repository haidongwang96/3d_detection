import cv2
import numpy as np

CONF_THRESHOLD = 0.5  # 关键点置信度阈值
MAX_AGE = 5  # 追踪器最大存活时间（帧数）


class KeyPointTracker:
    def __init__(self, person_id):
        self.person_id = person_id
        self.keypoints = {}  # {关键点ID: 卡尔曼滤波器}
        self.age = 0  # 未更新帧数计数器
        self.max_age = MAX_AGE

    def update_keypoint(self, kp_id, x, y, conf):
        """更新或初始化关键点追踪器"""
        if conf < CONF_THRESHOLD:
            return

        if kp_id not in self.keypoints:
            self._init_kalman(kp_id, x, y)
        else:
            self._update_kalman(kp_id, x, y)

        self.age = 0  # 重置未更新计数器

    def predict(self):
        """预测所有关键点的下一帧位置"""
        for kf in self.keypoints.values():
            kf.predict()
        self.age += 1

    def get_keypoints(self):
        """获取当前所有关键点预测位置"""
        return {
            kp_id: (kf.statePost[0, 0], kf.statePost[1, 0])
            for kp_id, kf in self.keypoints.items()
        }

    def _init_kalman(self, kp_id, x, y):
        """初始化卡尔曼滤波器"""
        kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵（匀速模型）
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 观测矩阵
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 初始化状态
        kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        # 噪声协方差矩阵（需要根据实际情况调整）
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        self.keypoints[kp_id] = kf

    def _update_kalman(self, kp_id, x, y):
        """用观测值更新卡尔曼滤波器"""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.keypoints[kp_id].correct(measurement)


class KeypointTrackerManager:
    def __init__(self):
        self.trackers = {}  # {人员ID: KeyPointTracker}

    def update(self, detections):
        """更新所有追踪器状态"""
        # 第一步：预测并清理过期追踪器
        self._predict_and_prune()

        # 第二步：处理新检测结果
        for person_id, kp_dict in detections:
            if person_id not in self.trackers:
                self.trackers[person_id] = KeyPointTracker(person_id)

            tracker = self.trackers[person_id]
            for kp_id, (x, y, conf) in kp_dict.items():
                tracker.update_keypoint(kp_id, x, y, conf)

    def get_keypoints(self):
        """获取所有追踪结果"""
        return {
            person_id: tracker.get_keypoints()
            for person_id, tracker in self.trackers.items()
        }

    def _predict_and_prune(self):
        """预测所有追踪器位置并删除过期追踪器"""
        # 预测所有现有追踪器
        for tracker in self.trackers.values():
            tracker.predict()

        # 删除过期追踪器
        expired_ids = [
            pid for pid, t in self.trackers.items()
            if t.age > t.max_age
        ]
        for pid in expired_ids:
            del self.trackers[pid]


# 使用示例
if __name__ == "__main__":
    # 初始化追踪管理器
    tracker_manager = KeypointTrackerManager()

    # 模拟连续帧的检测结果
    frame_detections = [
        # 第一帧：检测到ID=1的人员，包含两个关键点
        (1, {
            0: (100, 100, 0.9),  # 关键点0，高置信度
            1: (150, 150, 0.3)  # 关键点1，低置信度
        }),
        # 第二帧：检测到ID=1的人员，关键点0丢失
        (1, {
            0: (0, 0, 0.2),  # 低置信度
            1: (160, 160, 0.8)  # 关键点1，高置信度
        }),
        # 第三帧：ID=1的人员完全丢失
        # 第四帧：检测到ID=1的人员重新出现
        (1, {
            0: (120, 120, 0.9),
            1: (170, 170, 0.7)
        })
    ]

    for frame_idx, detections in enumerate(frame_detections):
        tracker_manager.update([detections])
        results = tracker_manager.get_keypoints()

        print(f"\nFrame {frame_idx + 1} 追踪结果:")
        for person_id, kps in results.items():
            print(f"Person {person_id}:")
            for kp_id, (x, y) in kps.items():
                print(f"  关键点 {kp_id}: ({x:.1f}, {y:.1f})")