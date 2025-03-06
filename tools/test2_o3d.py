
import cv2
import os
import time
import open3d as o3d
import numpy as np
import ultralytics
import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching

import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import camera
import model_tools as ml
import utility as su

os.makedirs("../plt", exist_ok=True)


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


class TrackHPObj:

    def __init__(self, track, obj:su.HumanPoseObject):
        self.track = track
        self.obj = obj


def process_predict(frame, predict, tracker):
    """
    处理yolo result结果
    :return:
    """
    boxes = predict.boxes.data.cpu().numpy()
    keypoints = predict.keypoints.data.cpu().numpy()
    # 提取yolo predict结果至humanposeobject
    # 注意predict为单相机（单图）结果，可能存在多个det结果
    objs = [su.HumanPoseObject(box, kypts) for box, kypts in zip(boxes, keypoints)]
    bboxs_traker = [obj.region2trackerformat for obj in objs]


    # reid  feature test
    # crop img 基于det 结果
    # crops = [obj.region.roi(frame) for obj in objs]
    # features = EXTRACTOR(crops).cpu()
    # deepsort
    # tracks = tracker.update_tracks(bboxs_traker, embeds=features)  # 2d tracking

    tracks = tracker.update_tracks(bboxs_traker, frame=frame)
    regions_t = [su.BoundingBox(*track.to_tlbr()) for track in tracks if track.is_confirmed()]
    regions_obj = [obj.region for obj in objs]
    track_objs = []
    if len(regions_t) != 0 and len(regions_obj) != 0:
        matches = ml.match_regions(regions_t, regions_obj, thresh=0.5) # todo: thresh 到底取多少？
        for match in matches:
            t_id, obj_id = match
            trackobj = TrackHPObj(tracks[t_id], objs[obj_id]) # merge track info and obj info
            track_objs.append(trackobj)

    return tracks, track_objs

def main():

    config = su.read_yaml_file('../data/record/office_3rd_floor_whd/config.yaml')
    CAM_IDS = config["cam_ids"]
    landmarks = camera.VirtualFences(config).landmarks

    multi_cam = camera.MultiCameraCapture(config)
    multi_cam.video_feeds("1732606939887")

    # 相机triangulation需要projection matrix
    P0 = multi_cam.Cameras[0].projection_matrix
    P1 = multi_cam.Cameras[1].projection_matrix

    # o3d 控制相机视角
    intr_mat = multi_cam.Cameras[0].intr.get_cam_mtx()
    extr_mat = multi_cam.Cameras[0].extr.pose_mat

    intrinsic = o3d.camera.PinholeCameraIntrinsic(500, 300, intr_mat)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extr_mat


    landmark3ds = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    landmark2d_xy = np.array([0,0,1,1]) # xymin, xymax
    boundary = [(-1, 2),(-1, 2),(0, 2)]
    landmark_conn = [[0,1],[1,2],[2,3],[3,0]]
    green = [0,1,0]  # RGB
    red = [1,0,0]

    # 初始化可视化窗口
    vis = su.PointcloudVisualizer(camera_params)
    single_o3d_obj = su.SingleKypto3d(vis)

    landmark_lineset = o3d.geometry.LineSet()
    landmark_lineset.points = o3d.utility.Vector3dVector(landmark3ds)
    landmark_lineset.lines = o3d.utility.Vector2iVector(landmark_conn)
    landmark_lineset.colors = o3d.utility.Vector3dVector([green for _ in landmark_conn])
    vis.add_geometry(landmark_lineset)


    TrackERS_2D = [DeepSort(n_init=3,
                            max_age=15,
                            max_iou_distance=0.9,
                            nms_max_overlap=1.0,
                            embedder="mobilenet",) for _ in CAM_IDS]
    skip = 0

    try:
        while True:
            t0 = time.time()
            frames = multi_cam.read_frames()
            # 录制的视频前面有跳帧，原因不明
            if skip < 10:
                skip += 1
                continue
            predicts = DETECTOR(frames, half=False, conf=0.5, iou=0.7, verbose=False)
            drawed_frames = []
            trackobjs_container = []
            #feature_dbs = [{} for _ in CAM_IDS]

            for i, predict in enumerate(predicts):
                if len(predict) == 0: # yolo 未检测到任何人
                    drawed_frames.append(frames[i])
                    continue
                #frames[i] = ml.predict_plot(frames[i],predict,draw_pose=True)
                tracks, trackobjs = process_predict(frames[i], predict, TrackERS_2D[i])
                if len(trackobjs) !=0:
                    trackobjs_container.append(trackobjs)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    #feature_dbs[i][track.track_id] = track.get_feature()
                    # todo: 画追踪轨迹
                    frames[i] = su.draw_track_bbox(frames[i], track.to_tlbr(), track.track_id, (255, 0, 0), thickness=2)

                for track_obj in trackobjs:
                        frames[i] = ml.obj_plot(frames[i], track_obj.obj, draw_bbox=True, draw_refined_bbox=False, draw_pose=True)
                drawed_frames.append(frames[i])

            #todo: 双相机 person reid
            # 假设配对成功，trackobjs_container的index为配对结果
            if len(trackobjs_container) >= 2:
                hpo3d = su.HumanPoseObject_3d([trackobj[0].obj for trackobj in trackobjs_container])
                hpo3d.valid_matched_kypts()
                kypt_pair = hpo3d.matched_kypts

                p3ds = camera.triangulation_cv2(P0, P1, kypt_pair[0],kypt_pair[1])
                p3ds_coco17 = su.p3d_2_kypt_17format(hpo3d.common_indices, p3ds)
                single_o3d_obj.update_data(p3ds_coco17)

                max_bound = np.asarray(single_o3d_obj.bbox.max_bound) # (x_max, y_max, z_max)
                min_bound = np.asarray(single_o3d_obj.bbox.min_bound) # (x_min, y_min, z_min)

                x_max, y_max = max_bound[:2]
                x_min, y_min = min_bound[:2]
                box_2d = [x_min, y_min, x_max, y_max]
                iou = su.roi2d(box_2d,landmark2d_xy)
                if iou > 0:
                    landmark_lineset.colors = o3d.utility.Vector3dVector([red for _ in landmark_conn])
                    step_in = True
                else:
                    landmark_lineset.colors = o3d.utility.Vector3dVector([green for _ in landmark_conn])
                    step_in = False

            else:
                single_o3d_obj.update_empty_points()
                landmark_lineset.colors = o3d.utility.Vector3dVector([green for _ in landmark_conn])
                step_in = False



            # if len(np.array(feature_dbs[0].values())) != 0 and len(np.array(feature_dbs[1].values())) != 0:
            #     res = nn_matching._nn_cosine_distance(np.array(feature_dbs[0]),np.array(feature_dbs[1]))
            #     print(res)
            #matches, m_scores = ml.match_features(feature_dbs[0], feature_dbs[1],threshold=0.65)
            if step_in:
                fence_color = (0,0,255)
            else:
                fence_color = (0,255,0)
            for i in range(len(drawed_frames)):
                su.draw_fence(drawed_frames[i], landmarks[i], color=fence_color)  # bgr

            vis.update(landmark_lineset)
            single_o3d_obj.update(vis)
            stack = np.vstack(drawed_frames)
            stack = cv2.resize(stack, (1280, 720*2))
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
