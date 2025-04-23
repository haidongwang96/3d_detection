# display only by cv2
# both 2d and 3d



import sys
import os
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径

import cv2
import time
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


angle_index_3d = 0
max_age= 5

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



def main():

    # 4. 初始化多相机捕获器
    with camera.MultiCameraCapture(config) as multi_cam:
        #multi_cam.video_feeds("1741249348001", video_folder="../data/record/tri_test/video_2")

        # single person
        #multi_cam.video_feeds("1741249303346", video_folder="video_0")
        #multi_cam.video_feeds("1741249326571", video_folder="video_1")
        #multi_cam.video_feeds("1741249348001", video_folder="video_2")


        # multi person
        #multi_cam.video_feeds("1741249374930", video_folder="../data/record/tri_test/video_3")
        multi_cam.video_feeds("1741249398397", video_folder="../data/record/tri_test/video_4")
        #multi_cam.video_feeds("1741249434256", video_folder="../data/record/tri_test/video_5")

        # todo: here !
        # landmarks = camera.VirtualFences(config).landmarks
        landmark_3 = [[591, 501],[457, 558],[603, 644],[729, 564]]
        landmark_5 = [[400, 433],[427, 546],[606, 544],[556, 445]]
        landmark_7 = [[659, 622],[790, 483],[622, 426],[475, 529]]
        landmarks = [landmark_3, landmark_5, landmark_7]


        # 6. 初始化跟踪器
        TrackERS_2D = [DeepSort(n_init=3,
                                max_age=15,
                                max_iou_distance=0.7,
                                nms_max_overlap=1.0,
                                embedder="mobilenet",) for _ in CAM_IDS]

        multi_tracker_3d = su.MultiHumanTracker3D_NoO3D(max_age=max_age)


        # 7. 主循环
        skip = 0
        try:
            while True:
                step_in = False
                frame_count = multi_cam._frame_count
                logging.info(f"frame_count: {frame_count}")
                t0 = time.time()
                # 读取视频帧
                frames = multi_cam.read_frames()
                # Draw 3D BBoxes onto 2D frames
                white_boards = [np.zeros_like(f)+255 for f in frames]
               
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
                        trackobjs_container.append([]) # Add empty list if no detections
                        continue

                    trackobjs = mt.process_predict(frames[i], predict, TrackERS_2D[i], config)

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

                    # Ensure container has an entry for each camera, even if empty
                    trackobjs_container.append(trackobjs)
                

                # Check if we have results for *enough* cameras (e.g., at least 2 for triangulation)
                # This logic might need adjustment based on how matches are handled with missing data
                valid_views_count = sum(1 for objs in trackobjs_container if objs)
                if valid_views_count >= 2: # Need at least two views for triangulation
                    # Potentially filter trackobjs_container and cam_ids/params passed to matching/update
                    # For now, assume matching handles potential empty lists if needed

                    # Find matches across cameras that have detections
                    matches = mt.improved_cross_camera_reid(trackobjs_container)

                    # 更新多人3D跟踪器
                    # Pass only relevant data if matching needs it
                    multi_tracker_3d.update(matches, trackobjs_container, cam_manager, CAM_IDS)

                    # 检查是否有人侵入围栏
                    step_in = multi_tracker_3d.check_fence_intrusion(landmark2d_xy)
                    logging.info(f"step_in: {step_in}")

                else:
                    # 无匹配时
                    multi_tracker_3d.update([], trackobjs_container, cam_manager, CAM_IDS)
                    #step_in = False
                
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
                        #logging.info(f"keypoints: {keypoints}")
                        #if bbox_3d:
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

                fps = 1 / (time.time() - t0) if (time.time() - t0) > 0 else 0
                cv2.putText(stack_all, f"FPS: {fps:.2f}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera View", stack_all)

                # 按下 'q' 键退出, waitKey(1) for video, waitKey(0) for stepping
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        finally: 
            multi_cam.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
