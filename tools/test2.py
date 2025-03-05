import cv2
import threading
from ultralytics import YOLO

import camera

# 加载YOLOv8模型
model = YOLO('yolov11n.pt')  # 你可以选择其他模型，如 'yolov8s.pt', 'yolov8m.pt' 等

# 定义相机处理线程
class CameraThread(threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.model = model
        self.frame = frame
        self.result = None

    def run(self):
        while self.running:
            self.result = self.model.track(self.frame, persist=True)


    def get_result(self):
        return self.result

    def stop(self):
        self.running = False

# 创建相机线程
camera_threads = []
v0 = "../data/record/office_3rd_floor_whd/1732606939887_2.mp4"
v1 = "../data/record/office_3rd_floor_whd/1732606939887_4.mp4"
multi_cam = camera.MultiCameraCapture([v0, v1])


while True:
    frames = multi_cam.read_frames()
    thread = CameraThread(frames)
    camera_threads.append(thread)
    thread.start()




#