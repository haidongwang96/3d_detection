import sys
import os
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径

import argparse
import cv2
import time
import datetime
import numpy as np
import ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import camera
import model_tools as mt
import utility as su


import logging
# 配置日志记录器


def run(config):

    # 4. 初始化多相机捕获器
    with camera.MultiCameraCapture(config) as multi_cam:
        multi_cam.camera_feeds(CAM_IDS)

        landmarks = []
        landmarks_dir = os.path.join(root, config['proj_root'], config['landmark'])
        #f"..data/record//{config['proj_root']}/{config['landmark']}"
        for cam_id in CAM_IDS:
            landmark_path = su.collect_file_by_index_prefix(landmarks_dir, cam_id, "txt")
            assert len(landmark_path) == 1, f"landmark_path: {landmark_path}"
            landmark_path = landmark_path[0]
            landmark_i = su.read_list_file(landmark_path," ")
            landmark_i = np.array(landmark_i, dtype=np.int32)
            landmarks.append(landmark_i)


        # 6. 初始化跟踪器
        TrackERS_2D = [DeepSort(n_init=tracker_2d_config['n_init'],
                                max_age=tracker_2d_config['max_age'],
                                max_iou_distance=tracker_2d_config['max_iou_distance'],
                                nms_max_overlap=tracker_2d_config['nms_max_overlap'],
                                embedder=tracker_2d_config['embedder'],) for _ in CAM_IDS]

        multi_tracker_3d = su.MultiHumanTracker3D_NoO3D(max_age=tracker_3d_config['max_age'])


        # 7. 主循环
        current_datetime = "None"
        start = time.time()
        # 添加变量用于隔帧处理
        frame_to_process = True
        # 存储上一帧处理结果
        last_trackobjs_container = []
        last_matches = []
        last_step_in = False
        # FPS计算相关变量
        fps_smoothing = 0.9  # 平滑系数，值越大平滑效果越强
        avg_fps = 0
        prev_time = time.time()
        try:
            while True:
                frame_count = multi_cam._frame_count
                logging.info(f"frame_count: {frame_count}")
                t0 = time.time()
                # 读取视频帧
                frames = multi_cam.read_frames()
                if frames is None:
                    break

                # Draw 3D BBoxes onto 2D frames
                white_boards = [np.zeros_like(f)+255 for f in frames]
               
                # 仅在需要处理的帧上执行检测和分析
                if frame_to_process:
                    step_in = False
                    # YOLO检测
                    predicts = DETECTOR(frames, half=yolo_config['half'], conf=yolo_config['conf'], iou=yolo_config['iou'], verbose=False)
                    trackobjs_container = []
                    for i, predict in enumerate(predicts):
                        if len(predict) == 0: # yolo 未检测到任何人
                            trackobjs_container.append([]) # Add empty list if no detections
                            continue

                        trackobjs = mt.process_predict(frames[i], predict, TrackERS_2D[i], config)

                        # 基于tracking+obj
                        for track_obj in trackobjs:
                            # 计算det底边中点 z=0 的3d点
                            p3d_z0 = track_obj.obj_floor_coordinate(Ks[i], Rs[i], ts[i])
                            track_obj.p3d_z0_det = np.array(p3d_z0)

                        # Ensure container has an entry for each camera, even if empty
                        trackobjs_container.append(trackobjs)
                    
                    # Check if we have results for *enough* cameras (e.g., at least 2 for triangulation)
                    valid_views_count = sum(1 for objs in trackobjs_container if objs)
                    if valid_views_count >= 2: # Need at least two views for triangulation
                        # Find matches across cameras that have detections
                        matches = mt.improved_cross_camera_reid(trackobjs_container)

                        # 更新多人3D跟踪器
                        multi_tracker_3d.update(matches, trackobjs_container, cam_manager, CAM_IDS)

                        # 检查是否有人侵入围栏
                        step_in = multi_tracker_3d.check_fence_intrusion(landmark2d_xy)
                        logging.info(f"step_in: {step_in}")
                    else:
                        # 无匹配时
                        matches = []
                        multi_tracker_3d.update([], trackobjs_container, cam_manager, CAM_IDS)
                    
                    # 保存当前处理结果，供下一帧使用
                    last_trackobjs_container = trackobjs_container
                    last_matches = matches
                    last_step_in = step_in
                else:
                    # 使用上一帧的处理结果
                    trackobjs_container = last_trackobjs_container
                    matches = last_matches
                    step_in = last_step_in
                
                # 绘制处理结果（无论是当前帧还是上一帧的结果）
                # 对每个相机视图绘制2D检测结果
                for i in range(len(frames)):
                    if i < len(trackobjs_container):
                        for track_obj in trackobjs_container[i]:
                            # 画出detetion 结果的框
                            frames[i] = mt.obj_plot(frames[i], track_obj.obj,
                                                    draw_bbox=True, draw_refined_bbox=False, draw_pose=True)

                            # 画出tracker的框，蓝色框
                            track = track_obj.track
                            if track.is_confirmed():
                                frames[i] = su.draw_track_bbox(frames[i], track.to_tlbr(), track.track_id, (255, 0, 0), thickness=2)

                            # 画出底边中点
                            cv2.circle(frames[i], track_obj.get_track_box_bottom_center, 2, (0, 165, 255), thickness=2) # 橙色
                            cv2.circle(frames[i], track_obj.get_obj_box_bottom_center, 2, (128, 0, 128), thickness=2) # 紫色

                # 更新所有跟踪对象状态 (like bbox calculation)
                for i in range(len(white_boards)): # Iterate through each camera view
                    k = Ks[i]
                    r = Rs[i]
                    t = ts[i]
                    rvec, _ = cv2.Rodrigues(r) # Convert rotation matrix to rvec for projectPoints

                    for track_id, track_data in multi_tracker_3d.tracks.items():
                        t_color = track_colors[int(track_id) % len(track_colors)]
                        bbox_3d = track_data.get('bbox') # [min_x, min_y, min_z, max_x, max_y, max_z]
                        keypoints = track_data.get('keypoints') # [x,y,z]
                        last_measured_frame = track_data.get('last_measured_frame')
                        age = multi_tracker_3d.frame_count - last_measured_frame
                        logging.info(f"bbox_3d: {bbox_3d}")

                        # Project 3D bbox vertices to 2D image plane
                        p2d = su.project_8_vertices_to_imgpoint(
                            bbox_3d[0], bbox_3d[3], bbox_3d[1], bbox_3d[4], bbox_3d[2], bbox_3d[5],
                        k, rvec, t
                        )
                        # 如果跟踪对象存在时间大于0，则显示为红色
                        if age > 0:
                            logging.info(f"color change,age: {age}")
                            t_color = (0, 0, 255)
                        # Draw the projected cube
                        white_boards[i] = su.drawCube(white_boards[i], p2d, color=t_color, thickness=2) 
                        # 在立方体上方显示track_id
                        # 获取立方体的顶部中心点作为文本位置
                        top_center = (p2d[4] + p2d[5] + p2d[6] + p2d[7]) / 4
                        text_pos = (int(top_center[0]), int(top_center[1] - 10))  # 在顶部中心点上方10像素
                        cv2.putText(white_boards[i], f"ID: {track_id}, age: {age}", text_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 2)
                        
                        # Project 3D keypoints to 2D image plane
                        p2d = su.project_keypoints_to_imgpoint(keypoints, k, rvec, t)
                        white_boards[i] = su.draw_keypoints_2d(white_boards[i], p2d)

                # 判断是否进入围栏
                fence_color = (0,0,255) if step_in else (0,255,0) # BGR
                for i in range(len(frames)):
                    su.draw_fence(frames[i], landmarks[i], color=fence_color)  # bgr
                    su.draw_fence(white_boards[i], landmarks[i], color=fence_color)  # bgr

                # 显示视频帧
                stack = np.vstack(frames)
                stack_white = np.vstack(white_boards)
                # Resize based on the number of cameras
                target_height = 480 * len(CAM_IDS)
                target_width = int(stack.shape[1] * (target_height / stack.shape[0])) # Maintain aspect ratio
                stack = cv2.resize(stack, (target_width, target_height))
                stack_white = cv2.resize(stack_white, (target_width, target_height))

                stack_all = np.hstack([stack, stack_white])
                if step_in:
                    # 获取 stack_all 的尺寸
                    h, w, _ = stack_all.shape
                    # 定义边框颜色 (红色 BGR) 和厚度
                    border_color = (0, 0, 255)
                    border_thickness = 10  # 可以根据需要调整边框厚度
                    # 在 stack_all 图像上绘制红色边框
                    cv2.rectangle(stack_all, (0, 0), (w - 1, h - 1), border_color, border_thickness)
                    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 计算平滑的FPS，考虑到隔帧处理
                current_fps = 1 / (time.time() - t0) if (time.time() - t0) > 0 else 0
                if avg_fps == 0:
                    avg_fps = current_fps
                else:
                    avg_fps = fps_smoothing * avg_fps + (1 - fps_smoothing) * current_fps
                
                cv2.putText(stack_all, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                current_time = time.time() - start
                minutes = int(current_time // 60)
                seconds = int(current_time % 60)
                current_time = f"{minutes:02d}:{seconds:02d}"
                cv2.putText(stack_all, f"Time: {current_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(stack_all, f"Last Intrusion: {current_datetime}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Camera View", stack_all)

                # 切换处理标志，实现隔帧处理
                frame_to_process = not frame_to_process

                # 按下 'q' 键退出, waitKey(1) for video, waitKey(0) for stepping
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally: 
            cv2.destroyAllWindows()
            


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
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
    track_colors = su.generate_distinct_colors(100)

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="G:/code/3d_detection/data/record/video_test_0/config.yaml")
    args = args.parse_args()


    # 1. 初始化配置
    config = su.read_yaml_file(args.config)
    root=config['root']
    CAM_IDS = config["cam_ids"]
    yolo_config = config["yolo"]
    tracker_2d_config = config["tracker_2d"]
    tracker_3d_config = config["tracker_3d"]


    # 2. 初始化相机管理器（处理相机参数）
    cam_manager = camera.CameraManager(config)

    # 3. 获取相机参数
    Ks, Rs, ts, Ps = [], [], [], []
    for id in cam_manager.cam_ids:
        K, R, t = cam_manager.get_camera_params(id)
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
        P = cam_manager.get_projection_matrix(id)
        Ps.append(P)

    DETECTOR = ultralytics.YOLO(os.path.join(root, yolo_config['model']))


    run(config)
