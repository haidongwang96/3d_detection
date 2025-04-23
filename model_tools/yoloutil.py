import numpy as np
import cv2
import scipy
from deep_sort_realtime.deep_sort import nn_matching

import utility as su
import camera
import logging


def process_predict(frame, predict, tracker, config):
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
    matches = match_regions_hybrid(
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

def match_regions_hybrid(track_regions: list[su.BoundingBox], det_regions: list[su.BoundingBox], 
                        max_distance: float = 50.0, min_iou: float = 0.3,
                        distance_weight: float = 0.7) -> list[tuple[int, int]]:
    """
    使用中心点距离和IoU的加权组合进行匹配。
    
    Args:
        track_regions: 跟踪器的边界框列表
        det_regions: 检测器的边界框列表
        max_distance: 最大允许的中心点距离
        min_iou: 最小IoU阈值
        distance_weight: 距离在最终得分中的权重(0-1)
        
    Returns:
        list[tuple[int, int]]: 匹配结果
    """
    def get_center_point(bbox: su.BoundingBox) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(center1: tuple[float, float], 
                         center2: tuple[float, float]) -> float:
        x1, y1 = center1
        x2, y2 = center2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def calculate_score(distance: float, iou: float) -> float:
        """计算综合得分"""
        # 将距离归一化到0-1范围
        norm_distance = 1 - min(distance / max_distance, 1)
        # 加权组合
        return distance_weight * norm_distance + (1 - distance_weight) * iou
    
    matches = []
    used_indices = set()
    
    # 预计算中心点
    track_centers = [get_center_point(region) for region in track_regions]
    det_centers = [get_center_point(region) for region in det_regions]
    
    # 对每个跟踪框寻找最佳匹配
    for i, track_center in enumerate(track_centers):
        best_score = 0
        best_idx = -1
        
        for j, det_center in enumerate(det_centers):
            if j in used_indices:
                continue
            
            # 计算距离和IoU
            distance = calculate_distance(track_center, det_center)
            iou = track_regions[i].iou(det_regions[j])
            
            # 如果IoU太小或距离太大，直接跳过
            if iou < min_iou or distance > max_distance:
                continue
            
            # 计算综合得分
            score = calculate_score(distance, iou)
            
            if score > best_score:
                best_score = score
                best_idx = j
        
        if best_idx != -1:
            matches.append((i, best_idx))
            used_indices.add(best_idx)
    
    return matches


def match_regions_greedy(track_regions:list[su.BoundingBox], det_regions:list[su.BoundingBox], thresh=0.7):
    """
    贪心算法

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

def match_region_hungarian(track_regions:list[su.BoundingBox], det_regions:list[su.BoundingBox], thresh=0.7):
    """
    在目标数量不超过5个的情况下比较匈牙利算法和贪心算法的效率：
    时间复杂度比较
    贪心算法：O(n²)，其中n是目标数量
    匈牙利算法：O(n³)，其中n是目标数量
    当n≤5时的实际效率
    对于n=5的情况：
    贪心算法：需要执行约25次操作（5×5）
    匈牙利算法：需要执行约125次操作（5³）
    实际运行效率分析
    计算量：
    贪心算法计算量明显更小
    对于n=5，贪心算法比匈牙利算法快约5倍
    常数因子：
    贪心算法的实现更简单，常数因子更小
    匈牙利算法需要额外的矩阵操作和复杂的优化步骤
    内存使用：
    贪心算法内存需求更小
    匈牙利算法需要存储完整的成本矩阵和中间计算结果
    结论
    在目标数量不超过5个的情况下：
    贪心算法的效率明显高于匈牙利算法
    贪心算法的实现更简单，运行更快，内存占用更少
    匈牙利算法的优势（全局最优解）在目标数量少时体现不明显
    因此，如果您的应用场景中目标数量通常不超过5个，并且对实时性要求较高，贪心算法是更好的选择。只有当您特别需要保证全局最优匹配，且可以接受一定的性能损失时，才建议使用匈牙利算法。


    将第二组 bbox 配队到第一组 bbox, 使用匈牙利算法进行最优匹配
    :param track_regions: 第一组 bbox, 跟踪得到的边界框
    :param det_regions: 第二组 bbox, 检测得到的边界框
    :param thresh: IoU 阈值，低于此阈值的匹配将被忽略
    :return: 配对结果, 列表中的每个元素是一个元组 (idx1, idx2), 表示 track_regions[idx1] 和 det_regions[idx2] 配对
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    if not track_regions or not det_regions:
        return []
    
    # 构建成本矩阵（负IoU，因为匈牙利算法是最小化成本）
    cost_matrix = np.zeros((len(track_regions), len(det_regions)))
    
    # 填充成本矩阵
    for i, region1 in enumerate(track_regions):
        for j, region2 in enumerate(det_regions):
            iou = region1.iou(region2)
            # 使用负IoU作为成本（因为我们想要最大化IoU）
            cost_matrix[i, j] = -iou
    
    # 使用匈牙利算法找到最优匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 过滤掉IoU低于阈值的匹配
    matches = []
    for row, col in zip(row_indices, col_indices):
        if -cost_matrix[row, col] > thresh:  # 转换回IoU值并检查阈值
            matches.append((row, col))
    
    return matches

def cross_camera_reid(trackobjs_list, alpha=0.3, feature_threshold=0.2, distance_threshold=1.5):
    """
    跨摄像头行人重识别
    
    Args:
        trackobjs_list: 各摄像头的跟踪对象列表
        alpha: 特征距离的权重
        beta: 地面距离的权重
        feature_threshold: 特征距离阈值
        distance_threshold: 地面距离阈值
        
    Returns:
        matches: 匹配结果列表，每个元素为 (cam1_idx, obj1_idx, cam2_idx, obj2_idx, score)
    """
    if len(trackobjs_list) < 2:
        return []
    logging.info(f"cam1_tracked num: {len(trackobjs_list[0])}, cam2_tracked num: {len(trackobjs_list[1])}")

    matches = []
    # 遍历所有摄像头对
    for i in range(len(trackobjs_list)):
        for j in range(i+1, len(trackobjs_list)):
            cam1_objs = trackobjs_list[i]
            cam2_objs = trackobjs_list[j]

            # 遍历摄像头i的所有对象
            for obj1 in cam1_objs:
                best_match = None
                best_score = float('inf')  # 越小越好
                
                # 获取特征
                feature1 = obj1.feature
                ground_point1 = obj1.p3d_z0_det
                track_id1 = obj1.track.track_id

                # 遍历摄像头j的所有对象
                for obj2 in cam2_objs:
                    # 获取特征
                    feature2 = obj2.feature
                    ground_point2 = obj2.p3d_z0_det
                    track_id2 = obj2.track.track_id
                    
                    # 计算特征距离 (欧氏距离)
                    # 对于特征距离，目前看来，余弦距离比欧氏距离更合适
                    # 但是，余弦距离在0-1之间，而欧氏距离在0-无穷大之间，所以需要归一化
                    # 余弦相似度的计算公式为：cosine_similarity = 1 - feature_distance
                    # 其中，feature_distance = 1 - np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))


                    
                    #feature_distance = np.linalg.norm(feature1 - feature2)
                    feature_cos_distance = 1 - np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
                    cosine_similarity = 1 - feature_cos_distance
                    
                    # 计算地面坐标距离
                    # 如果小于0.2，目前看来是非常小的，可以认为匹配上了
                    ground_distance = np.linalg.norm(ground_point1 - ground_point2)
                    logging.info(f"cam1_idx: {i}, track_id1: {track_id1} -- cam2_idx: {j}, track_id2: {track_id2}")
                    logging.info(f"feature_similarity: {cosine_similarity}, ground_distance: {ground_distance}")

                    score = cosine_similarity + ground_distance
                    
                    # 归一化距离 (可选)

                    
                    # 计算加权得分 (越小越好)
                    #beta = 1 - alpha
                    #score = alpha * norm_feature_dist + beta * norm_ground_dist
                    
                    # 更新最佳匹配
                    if score < best_score:
                        best_score = score
                        best_match = (i, track_id1, j, track_id2, score) # i, j 是摄像头索引，idx1, idx2 是对象（track)索引
                # 如果找到匹配且得分低于阈值，添加到结果中
                if best_match and best_score < 0.8:  # 0.8是综合阈值
                    matches.append(best_match)

            for match in matches:
                logging.info(f"cam1_idx: {match[0]}, track_id1: {match[1]} -- cam2_idx: {match[2]}, track_id2: {match[3]}, score: {match[4]}")

            logging.info(f"--------------------------------")
    
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



def improved_cross_camera_reid(trackobjs_list, similarity_weight=0.4, similarity_threshold=0.6, distance_threshold=1.5):
    """
    改进的跨摄像头行人重识别
    
    Args:
        trackobjs_list: 各摄像头的跟踪对象列表
        similarity_weight: 特征相似度的权重(0-1)
        similarity_threshold: 余弦相似度阈值
        distance_threshold: 地面距离阈值(米)
        
    Returns:
        matches: 匹配结果列表
    """
    if len(trackobjs_list) < 2:
        return []
    
    matches = []
    # 遍历所有摄像头对
    for i in range(len(trackobjs_list)):
        for j in range(i+1, len(trackobjs_list)):
            cam1_objs = trackobjs_list[i]
            cam2_objs = trackobjs_list[j]
            
            # 构建所有可能的匹配及其得分
            all_possible_matches = []
            
            for idx1, obj1 in enumerate(cam1_objs):
                feature1 = obj1.feature
                ground_point1 = obj1.p3d_z0_det
                track_id1 = obj1.track.track_id
                
                for idx2, obj2 in enumerate(cam2_objs):
                    feature2 = obj2.feature
                    ground_point2 = obj2.p3d_z0_det
                    track_id2 = obj2.track.track_id
                    
                    # 计算余弦相似度 (越大越好)
                    cosine_similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
                    
                    # 计算3D空间距离 (越小越好)
                    spatial_distance = np.linalg.norm(ground_point1 - ground_point2)
                    
                    # 归一化空间距离 (转换为0-1，越大越好)
                    norm_distance = max(0, 1 - spatial_distance / distance_threshold)
                    
                    # 计算综合得分 (越大越好)
                    score = similarity_weight * cosine_similarity + (1 - similarity_weight) * norm_distance
                    
                    # 只考虑相似度和距离都满足阈值的匹配
                    if cosine_similarity >= similarity_threshold and spatial_distance <= distance_threshold:
                        all_possible_matches.append((idx1, idx2, score, track_id1, track_id2))
            
            # 按得分降序排序
            all_possible_matches.sort(key=lambda x: x[2], reverse=True)
            
            # 贪心匹配，确保每个track_id只匹配一次
            matched_cam1 = set()
            matched_cam2 = set()
            
            for idx1, idx2, score, track_id1, track_id2 in all_possible_matches:
                if idx1 in matched_cam1 or idx2 in matched_cam2:
                    continue
                
                matches.append((i, idx1, j, idx2, score))
                matched_cam1.add(idx1)
                matched_cam2.add(idx2)
                
                # 记录日志
                logging.info(f"匹配成功: 摄像头{i}-{idx1}的track_id {track_id1} 与 摄像头{j}-{idx2}的track_id {track_id2}, 得分: {score:.4f}")
                logging.info(f"--------------------------------")

    return matches 