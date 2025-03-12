import sys
import os
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径

import cv2
import time
import open3d as o3d
import numpy as np
import ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import camera
import model_tools as mt
import utility as su

import logging
# 配置日志记录器
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # 将日志写入文件
                        logging.StreamHandler()  # 同时输出到控制台
                    ])


# 地标， 默认z轴=0， 单位长度为1
landmark3ds = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
landmark2d_xy = np.array([0 ,0, 1, 1]) # xymin, xymax
boundary = [(-1, 2),(-1, 2),(0, 2)]
landmark_conn = [[0,1],[1,2],[2,3],[3,0]]
green = [0,1,0]  # RGB
red = [1,0,0]
orange = [1, 0.647, 0]  
purple = [0.5, 0, 0.5]


# 1. 初始化配置
config = su.read_yaml_file('../data/record/tri_test/config.yaml')
CAM_IDS = config["cam_ids"]

# 2. 初始化相机管理器（处理相机参数）
cam_manager = camera.CameraManager('../data/record/tri_test/config.yaml')

# 3. 获取相机参数
Ks, Rs, ts, Ps = [], [], [], []
for id in cam_manager.cam_ids:
    K, R, t = cam_manager.get_camera_params(id)
    Ks.append(K)
    Rs.append(R)
    ts.append(t)
    P = cam_manager.get_projection_matrix(id)
    Ps.append(P)



DETECTOR = ultralytics.YOLO("../data/weights/yolo11n-pose.pt")
# EXTRACTOR = torchreid.utils.FeatureExtractor(
#     model_name='mobilenetv2_x1_0',
#     model_path="../data/weights/mobilenetv2_1dot0_market .pth.tar",
#     device='cuda'
# )
# EXTRACTOR = torchreid.utils.FeatureExtractor(
#     model_name='osnet_x1_0',
#     model_path="../data/weights/osnet_x1_0_imagenet.pth",
#     device='cuda'
# )


def process_predict(frame, predict, tracker):
    """
    处理单个相机视角的 YOLO 检测结果，并与跟踪器结合。
    
    流程：
    1. 提取检测结果（边界框和关键点）
    2. 更新跟踪器
    3. 匹配跟踪结果和检测结果
    4. 融合跟踪和姿态信息
    
    Args:
        frame (np.ndarray): 当前帧图像
        predict (ultralytics.yolo.engine.results.Results): YOLO 检测结果
        tracker (DeepSort): 目标跟踪器实例
        
    Returns:
        tuple: (所有跟踪对象, 匹配的跟踪姿态对象)
    """
    # 步骤 1: 提取检测结果
    # ----------------------------------------
    # 从 YOLO 结果中提取边界框和关键点
    boxes = predict.boxes.data.cpu().numpy()  # 形状: (N, 6) [x1,y1,x2,y2,conf,cls]
    keypoints = predict.keypoints.data.cpu().numpy()  # 形状: (N, K, 3) [x,y,conf]
    
    # 创建人体姿态对象列表
    detection_objects = [
        su.HumanPoseObject(box, keypoints) 
        for box, keypoints in zip(boxes, keypoints)
    ]
    
    # 转换为跟踪器所需的格式
    tracker_inputs = [obj.region2trackerformat for obj in detection_objects]
    
    
    # 当前使用基本跟踪
    tracks = tracker.update_tracks(tracker_inputs, frame=frame)
    
    # 步骤 3: 匹配跟踪结果和检测结果

    # 筛选已确认的跟踪对象
    confirmed_tracks = [track for track in tracks if track.is_confirmed()]
    
    # 如果没有跟踪或检测，直接返回
    if not confirmed_tracks or not detection_objects: 
        return []
    
    # 转换为统一的边界框格式用于匹配
    track_regions = [su.BoundingBox(*track.to_tlbr()) for track in confirmed_tracks]
    detection_regions = [obj.region for obj in detection_objects]
    
    # 使用贪心算法匹配跟踪和检测
    # 注: 未来可以替换为匈牙利算法或其他更高级的匹配方法
    matches = mt.match_regions_hybrid(
        track_regions, 
        detection_regions,
        max_distance=50.0,
        min_iou=0.35,
        distance_weight=0.7
    )

    
    # 步骤 4: 融合跟踪和姿态信息
    # ----------------------------------------
    track_pose_objects = []  
    for track_idx, detection_idx in matches:

        # 如果贴近底边，5pixel以内，则不匹配
        if detection_objects[detection_idx].region.y2 > config["cam_record"]["HEIGHT"] - 10 :
            #logging.info(f"{confirmed_tracks[track_idx].track_id} on bottom")
            continue

        # 创建融合了跟踪和姿态信息的对象
        track_obj = su.TrackHPObj(
            confirmed_tracks[track_idx], 
            detection_objects[detection_idx]
        )

        track_obj.feature = confirmed_tracks[track_idx].get_feature()
        
        # 存储用于后续处理的信息
        track_pose_objects.append(track_obj)

    return track_pose_objects

def main():

    # 4. 初始化多相机捕获器
    with camera.MultiCameraCapture(config) as multi_cam:
        #multi_cam.video_feeds("1741249348001", video_folder="../data/record/tri_test/video_2")

        # single person
        #multi_cam.video_feeds("1741249303346", video_folder="video_0")
        #multi_cam.video_feeds("1741249326571", video_folder="video_1")
        #multi_cam.video_feeds("1741249348001", video_folder="video_2")


        # multi person
        multi_cam.video_feeds("1741249374930", video_folder="../data/record/tri_test/video_3")
        # multi_cam.video_feeds("1741249398397", video_folder="video_4")
        # multi_cam.video_feeds("1741249434256", video_folder="../data/record/tri_test/video_5")

        # todo: here !
        # landmarks = camera.VirtualFences(config).landmarks
        landmark_3 = [[591, 501],[457, 558],[603, 644],[729, 564]]
        landmark_5 = [[400, 433],[427, 546],[606, 544],[556, 445]]
        landmark_7 = [[659, 622],[790, 483],[622, 426],[475, 529]]
        landmarks = [landmark_3, landmark_5, landmark_7]

        # 5. 初始化 Open3D 可视化
        # o3d 控制相机视角
        o3d_angle_index = 1
        intr_mat = cam_manager.intrinsics[CAM_IDS[o3d_angle_index]]
        extr_mat = cam_manager.extrinsics[CAM_IDS[o3d_angle_index]]['pose_mat']

        # o3d 相机拍摄传入参数
        intrinsic = o3d.camera.PinholeCameraIntrinsic(500, 300, intr_mat)
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic
        camera_params.extrinsic = extr_mat

        # 初始化可视化窗口
        vis = su.PointcloudVisualizer(camera_params)
        # 初始化多人3D跟踪器
        multi_tracker_3d = su.MultiHumanTracker3D(vis)

        # todo: 多个，加入主逻辑 
        # 初始化3d点云对象
        #single_o3d_obj = su.SingleKypto3d(vis)
 
        # 初始化地标线集
        landmark_lineset = o3d.geometry.LineSet()
        landmark_lineset.points = o3d.utility.Vector3dVector(landmark3ds)
        landmark_lineset.lines = o3d.utility.Vector2iVector(landmark_conn)
        landmark_lineset.colors = o3d.utility.Vector3dVector([green for _ in landmark_conn])
        landmark_name = vis.add_geometry(landmark_lineset, "landmark")


        # 6. 初始化跟踪器
        TrackERS_2D = [DeepSort(n_init=3,
                                max_age=15,
                                max_iou_distance=0.7,
                                nms_max_overlap=1.0,
                                embedder="mobilenet",) for _ in CAM_IDS]
        # 7. 主循环
        skip = 0
        try:
            while True:
                frame_count = multi_cam._frame_count
                logging.info(f"frame_count: {frame_count}")
                t0 = time.time()
                # 读取视频帧
                frames = multi_cam.read_frames()
               
                # 录制的视频前面有跳帧，原因不明
                # 处理跳帧
                if skip < 1:
                    skip += 1
                    continue

                # YOLO检测
                predicts = DETECTOR(frames, half=False, conf=0.5, iou=0.7, verbose=False)
                trackobjs_container = []
                for i, predict in enumerate(predicts):
                    if len(predict) == 0: # yolo 未检测到任何人
                        continue

                    trackobjs = process_predict(frames[i], predict, TrackERS_2D[i])

                    # 基于tracking+obj
                    for track_obj in trackobjs:
                        # 画出detetion 结果的框
                        frames[i] = mt.obj_plot(frames[i], track_obj.obj, 
                                                draw_bbox=True, draw_refined_bbox=False, draw_pose=True)
                        
                        # 画出tracker的框，蓝色框
                        track = track_obj.track
                        if track.is_confirmed():                   
                            frames[i] = su.draw_track_bbox(frames[i], track.to_tlbr(), track.track_id, (255, 0, 0), thickness=2)
                            # todo: 画追踪轨迹

                        # 画出底边中点
                        cv2.circle(frames[i], track_obj.get_track_box_bottom_center, 2, (0, 165, 255), thickness=2) # 橙色
                        cv2.circle(frames[i], track_obj.get_obj_box_bottom_center, 2, (128, 0, 128), thickness=2) # 紫色

                        # 计算det底边中点 z=0 的3d点
                        p3d_z0 = track_obj.obj_floor_coordinate(Ks[i], Rs[i], ts[i])
                        track_obj.p3d_z0_det = np.array(p3d_z0)
                        
                    if len(trackobjs) !=0:
                        trackobjs_container.append(trackobjs)
                

                if len(trackobjs_container) == len(CAM_IDS): #每个视角都有跟踪结果
                    # 更新3d 地面跟踪点        
                    p3d_z0_dets = [track_obj.p3d_z0_det for track_obj in trackobjs for trackobjs in trackobjs_container]
                    p3d_z0_dets = np.array(p3d_z0_dets).astype(np.float32)

                    matches = mt.improved_cross_camera_reid(trackobjs_container)
                    #logging.info(f"matches: {matches}")

                    # 更新多人3D跟踪器
                    multi_tracker_3d.update(matches, trackobjs_container, cam_manager, CAM_IDS)
                    
                    # 检查是否有人侵入围栏
                    step_in = multi_tracker_3d.check_fence_intrusion(landmark2d_xy)

                    # 更新围栏颜色
                    if step_in:
                        landmark_lineset.paint_uniform_color(red)
                    else:
                        landmark_lineset.paint_uniform_color(green)
                    
                    vis.update(landmark_name)
                    # 为每个跟踪对象绘制2D投影

                else:
                    step_in = False
                    landmark_lineset.paint_uniform_color(green)
                    vis.update(landmark_name)
                
                # 更新所有跟踪对象
                multi_tracker_3d.update_all()

                # 判断是否进入围栏  
                fence_color = (0,0,255) if step_in else (0,255,0)
                for i in range(len(frames)):
                    su.draw_fence(frames[i], landmarks[i], color=fence_color)  # bgr

                # 更新可视化
                vis.update(landmark_lineset)

                # 显示视频帧
                stack = np.vstack(frames)
                stack = cv2.resize(stack, (960, 480*len(CAM_IDS)))
                fps = 1 / (time.time() - t0)
                cv2.putText(stack, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera", stack)
                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            vis.destroy()
            multi_cam.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
