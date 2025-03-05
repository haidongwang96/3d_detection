import numpy as np
import cv2
import scipy
from deep_sort_realtime.deep_sort import nn_matching

import utility as su

verbose = False


def match_features(features_db1, features_db2, threshold=0.7):

    matches = []
    scores = []
    for id1, feat1 in features_db1.items():
        for id2, feat2 in features_db2.items():
            # euclidean_distances similarity
            similarity = nn_matching._nn_cosine_distance(feat1, feat2)
            #similarity= round(1 / (1 + np.linalg.norm(np.linalg.norm(feat1) - np.linalg.norm(feat2))),3)
            # cosine distance
            #similarity = np.dot(feat1, feat2.T) / (np.linalg.norm(feat1)*np.linalg.norm(feat2))
            scores.append([id1, id2, similarity])
            print(f"第一图中 track_id:{id1} 和 第二图中 track_id:{id2} 相似度为{similarity}")
            if similarity > threshold:
                matches.append((id1, id2, similarity))
    return matches, np.array(scores)


def match_regions(track_regions:list[su.BoundingBox], det_regions:list[su.BoundingBox], thresh=0.7):
    """
    将第二组 bbox 配队到第一组 bbox
    :param bboxes1: 第一组 bbox, 形状为 [N, 4], 每行表示一个 bbox [x1, y1, x2, y2]
    :param bboxes2: 第二组 bbox, 形状为 [M, 4], 每行表示一个 bbox [x1, y1, x2, y2]
    :param thresh: IoU 阈值
    :return: 配对结果, 列表中的每个元素是一个元组 (idx1, idx2), 表示 bboxes1[idx1] 和 bboxes2[idx2] 配对
    """
    matches = []
    used_indices = set()  # 记录已经配对的第二组 bbox 索引

    for i, region1 in enumerate(track_regions):
        best_iou = thresh
        best_idx = -1
        for j, region2 in enumerate(det_regions):
            if j in used_indices:
                continue  # 跳过已经配对的 bbox
            iou = region1.iou(region2)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx != -1:
            matches.append((i, best_idx))
            used_indices.add(best_idx)

    return matches

def predict_plot(frame, predict, draw_bbox=True, draw_pose=False, img_size=(720,1280),color=(0,255,0),thickness=2):
    # img_size : (h,w)

    ann = su.EasyAnnotator(frame)
    boxes = predict.boxes.data.cpu().numpy()
    kypts = predict.keypoints.data.cpu().numpy()

    for box, kypt in zip(boxes, kypts):
        x1,y1,x2,y2, conf, label = box
        if draw_bbox:
            ann.text_label((x1,y1,x2,y2),str(label),txt_color=color,thickness=thickness)
            ann.draw_box((x1,y1,x2,y2),color=color,thickness=thickness)
        if draw_pose:
            ann.kpts(kypt, shape=img_size)
    return ann.im

def obj_plot(frame, obj:su.HumanPoseObject,draw_bbox=True, draw_refined_bbox=False, draw_pose=False, img_size=(720,1280),color=(0,255,0),thickness=2):
    # img_size : (h,w))

    ann = su.EasyAnnotator(frame)
    if draw_bbox:
        ann.draw_box(obj.box, color=color, thickness=thickness)
    if draw_refined_bbox:
        refined_box = obj.track_item.to_tlbr()
        ann.text_label(refined_box, str(obj.track_item.track_id), txt_color=color, thickness=thickness)
        ann.draw_box(refined_box, color=color, thickness=thickness)
    if draw_pose:
        ann.kpts(obj.keypoints, shape=img_size)

    return ann.im