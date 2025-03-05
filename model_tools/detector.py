import os
import yaml
import torch
import numpy as np
import ultralytics

import utility as su


cfg = su.read_yaml_file("yolo_cfg")

class YoloDetectorV11:

    def __init__(self, checkpoint, args, device=0):
        self.device = device
        self.checkpoint = checkpoint

        self.model = ultralytics.YOLO(self.checkpoint)
        self.args = args



class YoloDetector:

    # Yolov5需要使用我们自己的版本.
    # input_size可以是tuple, 格式为: (height, width)
    def __init__(self, gpuid, checkpoint, input_size):
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        repo = os.path.join(torch.hub.get_dir(), "ultralytics_yolov5_master")
        checkpoint = su.normlize_path(checkpoint)
        self.device = torch.device(f"cuda:{gpuid}")
        self.model = torch.hub.load(
            repo,
            model="custom",
            source="local",
            path=checkpoint,
            device=self.device)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != env:
            su.log_and_die("official yolov5 is not supported.")
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.class_names = self.model.names
        # 这个model的最外层其实是AutoShape.
        # 模型参数参考models/common.py中的定义.
        self.model.agnostic = True
        self.batch_size = 1
        # 这个参数用作模型标识, 供外部使用
        self.name = None

    def update_class_names(self, id_name_map):
        # 模型导出到engine格式之后, class_names丢失, 这里手动补上
        id_name_map = {int(i): n for i, n in id_name_map.items()}
        self.class_names = id_name_map
        self.model.names = id_name_map

    def _forward_one_batch(self, images):
        # 这里image要转成RGB. 参考函数说明: AutoShape.forward().
        # 位于yolov5项目: models/common.py
        images = [img[:, :, ::-1] for img in images]
        predicts = self.model(images, size=self.input_size).pred
        predicts = [p.detach().cpu().tolist() for p in predicts]
        return predicts

    def _predicts2regions(self, predicts):
        regions = []
        for *bbox, score, label in predicts:
            label = self.class_names[int(label)]
            regions.append(su.Region(bbox, label, score).toint())
        # 采用agnostic模式会在nms中对overlap去重, 这里就不需要了
        # return su.remove_overlap_regions(regions, self.max_iou)
        return regions

    # 在这里改变batch_size的大小
    def warmup(self, batch_size):
        self.batch_size = batch_size
        fake = np.zeros((*self.input_size, 3), dtype=np.uint8)
        self._forward_one_batch([fake] * batch_size)

    def forward(self, images):
        if not images: return []
        batches = su.group_by_index(images, self.batch_size)
        results = [self._forward_one_batch(b) for b in batches]
        return [r for rs in results for r in rs]

    def process(self, images):
        if not images: return []
        predicts = self.forward(images)
        return [self._predicts2regions(pred) for pred in predicts]
