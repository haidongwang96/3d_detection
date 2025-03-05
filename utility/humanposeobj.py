import utility as su
import camera
import numpy as np


class HumanPoseObject:

    """
    使用 ultralytics版yolo predict结果传入
    适配boxes，和 keypoint(optional)
    保存单个检测框的信息，其中含有boxes，和keypoint（optional)
    """
    def __init__(self, boxes, keypoints=None):
        """
        :param boxes: predict.boxes.data.cpu().numpy()  (shape: (6,)   x1,y1,x2,y2,conf,cls
        :param keypoints: predict.keypoints.data.cpu().numpy() (shape: (17,3)   x,y,cls
        """
        self.boxes = boxes
        self.box = tuple(self.boxes[:4])
        self.keypoints = keypoints
        self.region = self.boxes2BoundingBox()

    def boxes2BoundingBox(self, expand=False):
        lbb = su.LabeledBoundingBox(bbox=self.box, label="person", score=self.boxes[4])
        if expand:
            lbb = lbb.expand(ratio_h=0.05, ratio_w=0.05)
        return lbb

    @property
    def region2trackerformat(self):
        """
            raw_detections (horizontal bb) : List[ Tuple[ List[float or int], float, str ] ]
        List of detections, each in tuples of ( [left,top,w,h] , confidence, detection_class)
        """
        x, y = self.region.top_left
        w, h = self.region.width, self.region.height
        return [x, y, w, h], self.region.score, self.region.label

    def keypoints_reform_by_thresh(self, threshold):
        # 对于关节点小于threshold的赋值0，大于赋值1
        data = self.keypoints.copy()
        data[:, 2] = (data[:, 2] >= threshold).astype(int)
        self.keypoints = data

    def valid_keypoints_extract(self, indices):
        return self.keypoints[indices,: 2]


class HumanPoseObject_3d:

    def __init__(self, objs:list[HumanPoseObject]):
        self.objs = objs
        self.common_indices = None
        self.matched_kypts = None


    def valid_matched_kypts(self):
        #
        self.multi_view_keypoints_reform_by_thresh()
        self.common_indices = self.extract_valid_kypts()
        temp = []
        for obj in self.objs:
            kypt = obj.valid_keypoints_extract(self.common_indices)
            temp.append(kypt)
        self.matched_kypts = np.array(temp)
        return self.matched_kypts

    def multi_view_keypoints_reform_by_thresh(self, threshold=0.5):
        # 将全部视角的keypoint全部进行reform，即删除conf<threshold的点
        for obj in self.objs:
            obj.keypoints_reform_by_thresh(threshold=threshold)

    def extract_valid_kypts(self):
        # 将valid配对（同一个关节）全部视角的keypoint，提取出关节号
        masks= []
        for obj in self.objs:
            mask_i = obj.keypoints[:,2] == 1
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

    obj1 = HumanPoseObject(example_bbox1,example_kypt1)
    obj2 = HumanPoseObject(example_bbox2,example_kypt2)

    obj_3d = HumanPoseObject_3d([obj1, obj2])
    kypts = obj_3d.valid_matched_kypts()
    print(obj_3d.common_indices)
    print(kypts.shape)










