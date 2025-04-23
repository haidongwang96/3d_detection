import os

import cv2
import open3d as o3d

import numpy as np
from filterpy.kalman import KalmanFilter

import logging

import utility as su

class Colors:
    """
    Ultralytics color palette https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.

    ## Ultralytics Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose Color Palette

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "Ultralytics Brand Colors"

        For Ultralytics brand colors see [https://www.ultralytics.com/brand](https://www.ultralytics.com/brand). Please use the official Ultralytics colors for all marketing materials.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

class skeleton_util():
    def __init__(self):
        self.skeleton = np.array([[16, 14],[14, 12],[17, 15],[15, 13],
                         [12, 13],[6, 12],[7, 13],[6, 7],[6, 8],
                         [7, 9],[8, 10],[9, 11],[2, 3],[1, 2],
                             [1, 3],[2, 4],[3, 5],[4, 6],[5, 7]])

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

sk_util = skeleton_util()

def duo_camera_pose_preprocess(kpts0, kpts1, conf_thres=0.5):
    # 去去除掉不符合conf threshold的点, 不能小于0.5
    nkpt0, ndim0 = kpts0.shape
    nkpt1, ndim1 = kpts1.shape

    is_pose = nkpt0 ==nkpt1 == 17 and ndim0 == ndim1 ==3
    #assert is_pose
    duo_pose=[]
    for i, (k0, k1) in enumerate(zip(kpts0, kpts1)):
        x0_coord, y0_coord, conf0 = k0[0], k0[1], k0[2]
        x1_coord, y1_coord, conf1 = k1[0], k1[1], k1[2]
        if conf0 < conf_thres or conf1 < conf_thres:
            continue
        duo_pose.append([i, x0_coord, y0_coord, x1_coord, y1_coord])

    return np.array(duo_pose)

def p3d_2_kypt_17format(index, p3ds):
    """
    此函数配合 duo_camera_pose_preprocess生成的index， 将pts转变成17*3的格式
    """
    kpts_3d = []
    for idx, p3d in zip(index, p3ds):
        idx = int(idx)
        while idx > len(kpts_3d):
            kpts_3d.append([None, None, None])
        kpts_3d.append(p3d)

    while len(kpts_3d) < 17:
        kpts_3d.append([None, None, None])

    return np.array(kpts_3d)



def pose_3d_plot(ax, p3ds):

    for i, k in enumerate(p3ds):
        x, y, z = k
        if x == y == z == 0:
            continue
        ax.scatter(x, y, z, color=sk_util.kpt_color[i]/255, marker='o')

    for i, sk in enumerate(sk_util.skeleton):
        pos0 = (p3ds[(sk[0] - 1), 0], p3ds[(sk[0] - 1), 1], p3ds[(sk[0] - 1), 2])
        pos1 = (p3ds[(sk[1] - 1), 0], p3ds[(sk[1] - 1), 1], p3ds[(sk[1] - 1), 2])
        if pos0[0] == pos0[1] == pos0[2] == 0 or pos1[0] == pos1[1] == pos1[2] == 0:
            continue
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], [pos0[2], pos1[2]],
                color=sk_util.limb_color[i]/255, linestyle='-')

    return ax

def draw_keypoints_2d(img, keypoints, conf_threshold=0.5, thickness=2):
    """
    Draws 2D keypoints and skeleton lines onto an OpenCV image.

    Args:
        img (np.ndarray): The OpenCV image frame (BGR).
        keypoints (np.ndarray): Array of 2D keypoints, shape (N, 2) or (N, 3).
                                If shape is (N, 3), the third element is confidence.
        conf_threshold (float): Minimum confidence threshold to draw a keypoint (if confidence is available).
        thickness (int): Thickness of the lines and keypoint circles.

    Returns:
        np.ndarray: The image with keypoints and skeleton drawn on it.
    """
    h, w, _ = img.shape
    drawn_points = []

    # Draw Keypoints
    for i, kpt in enumerate(keypoints):
        if kpt is None:
            drawn_points.append(None)
            continue
            
        x, y = int(round(kpt[0])), int(round(kpt[1]))
        conf = kpt[2] if len(kpt) == 3 else 1.0 # Assume confidence 1 if not provided

        if conf < conf_threshold or not (0 <= x < w and 0 <= y < h):
            drawn_points.append(None) # Skip low confidence or out-of-bounds points
            continue

        point = (x, y)
        drawn_points.append(point)
        color_bgr = tuple(map(int, sk_util.kpt_color[i][::-1])) # RGB to BGR
        cv2.circle(img, point, radius=thickness+1, color=color_bgr, thickness=-1) # Draw filled circle

    # Draw Limbs
    for i, sk in enumerate(sk_util.skeleton):
        idx0 = sk[0] - 1
        idx1 = sk[1] - 1

        pos0 = drawn_points[idx0] if idx0 < len(drawn_points) else None
        pos1 = drawn_points[idx1] if idx1 < len(drawn_points) else None

        if pos0 is not None and pos1 is not None:
            color_bgr = tuple(map(int, sk_util.limb_color[i][::-1])) # RGB to BGR
            cv2.line(img, pos0, pos1, color=color_bgr, thickness=thickness)

    return img

ldm_line= [[0,1],[1,2],[2,3],[3,0]]
def draw_fence(img, landmarks, color=(0, 255, 0), thickness=2):
    #  链接四个角点的virtual fence
    for landmark in landmarks:
        cv2.circle(img, (landmark[0], landmark[1]), 2, color, thickness)

    for i, line in enumerate(ldm_line):
        pt1 = tuple((landmarks[line[0]][0], landmarks[line[0]][1]))
        pt2 = tuple((landmarks[line[1]][0], landmarks[line[1]][1]))
        cv2.line(img, pt1, pt2, color, thickness)

    return img

def plot_3d_fence(ax, landmarks_3d, color):

    for i, k in enumerate(landmarks_3d):
        x, y, z = k
        if x == y == z == 0:
            continue
        ax.scatter(x, y, z, color=color, marker='o')

    for line in ldm_line:

        pos0 = tuple((landmarks_3d[line[0]][0], landmarks_3d[line[0]][1], landmarks_3d[line[0]][2]))
        pos1 = tuple((landmarks_3d[line[1]][0], landmarks_3d[line[1]][1], landmarks_3d[line[1]][2]))
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], [pos0[2], pos1[2]],
                color=color, linestyle='-')

    return ax

def roi2d(box1, box2):
    # xmin,ymin,xmax,ymax = box1
    # xmin_f,ymin_f,xmax_f,ymax_f = fence(box2)

    # 计算交集区域
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # 计算交集面积
    inter_width = max(0, xmax_inter - xmin_inter)
    inter_height = max(0, ymax_inter - ymin_inter)
    inter_area = inter_width * inter_height

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def filter_valid_kypt_for_o3d_input(p3d):
    # p3ds : (17,3)
    p3d_indices = np.where((p3d[:, 0] != 0) & (p3d[:, 1] != 0) & (p3d[:, 2] != 0))[0]
    p3d_colors = sk_util.kpt_color[p3d_indices] / 255
    p3d_conn = []
    p3d_conn_color = []
    for i, sk in enumerate(sk_util.skeleton):
        pos1 = int(sk[0]-1) # sk_util.skeleton 中index需要-1
        pos2 = int(sk[1]-1)
        if pos1 in p3d_indices and pos2 in p3d_indices:
            p3d_conn.append([sk[0]-1, sk[1]-1])
            p3d_conn_color.append(sk_util.limb_color[i]/255)

    return p3d_indices, p3d_colors, p3d_conn, p3d_conn_color



class PointcloudVisualizer():
    """改进的点云可视化器，确保几何体的添加、更新和移除操作正确同步"""

    def __init__(self, camera_params):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1280, height=720)
        self.camera_params = camera_params
        
        # 跟踪已添加的几何体
        self.geometries = {}
        self.ctr = self.vis.get_view_control()
        
        # 添加坐标系作为默认几何体，提供视觉参考
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # self.add_geometry(coord_frame, "coordinate_frame")
        
        # 添加一个网格平面作为地面参考
        # ground_plane = self._create_ground_plane(size=2.0, height=0.0)
        # self.add_geometry(ground_plane, "ground_plane")
        
        # 设置初始视图并立即更新渲染器

        self.vis.update_renderer()
        self.vis.poll_events()
        self.reset_view()
        
        logging.info("PointcloudVisualizer 初始化完成")

    def _create_ground_plane(self, size=2.0, height=0.0):
        """创建地面网格平面"""
        # 创建一个平面网格
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=0.001, depth=size)
        # 将平面移动到指定高度
        mesh.translate([-size/2, height, -size/2])
        # 设置颜色为浅灰色
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        return mesh

    def reset_view(self):
        """重置视图到初始相机参数"""
        # 尝试应用相机参数
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params, allow_arbitrary=True)
        #logging.info("成功应用相机参数")
        
    def add_geometry(self, cloud, name=None):
        """添加几何体并跟踪它"""
        if name is None:
            name = f"geometry_{len(self.geometries)}"
            
        if name in self.geometries:
            # 如果已存在同名几何体，先移除它
            self.vis.remove_geometry(self.geometries[name])
            
        self.geometries[name] = cloud
        success = self.vis.add_geometry(cloud)
        if not success:
            logging.warning(f"添加几何体 {name} 失败")
        else:
            logging.info(f"成功添加几何体 {name}")
        return name

    def remove_geometry(self, name_or_cloud):
        """移除几何体"""
        if isinstance(name_or_cloud, str):
            # 通过名称移除
            if name_or_cloud in self.geometries:
                success = self.vis.remove_geometry(self.geometries[name_or_cloud])
                if success:
                    del self.geometries[name_or_cloud]
                    return True
                else:
                    logging.warning(f"移除几何体 {name_or_cloud} 失败")
                    return False
            return False
        else:
            # 通过对象移除
            for name, geom in list(self.geometries.items()):
                if geom is name_or_cloud:
                    success = self.vis.remove_geometry(geom)
                    if success:
                        del self.geometries[name]
                        return True
                    else:
                        logging.warning(f"移除几何体 {name} 失败")
                        return False
            return False

    def update(self, name_or_cloud=None):
        """更新特定几何体或所有几何体"""
        if name_or_cloud is None:
            # 更新所有几何体
            for geom in self.geometries.values():
                self.vis.update_geometry(geom)
        elif isinstance(name_or_cloud, str):
            # 通过名称更新
            if name_or_cloud in self.geometries:
                self.vis.update_geometry(self.geometries[name_or_cloud])
        else:
            # 通过对象更新
            self.vis.update_geometry(name_or_cloud)
            
        # 更新渲染器和事件
        self.vis.update_renderer()
        self.vis.poll_events()
        self.reset_view()

    def destroy(self):
        """销毁窗口"""
        self.vis.destroy_window()


class SingleKypto3d:
    """改进的3D人体姿态可视化类，更好地处理空几何体"""

    green = [0,1,0]  # RGB
    red = [1,0,0]
    orange = [1, 0.647, 0]  
    purple = [0.5, 0, 0.5]
    blue = [0, 0, 1]
    white = [1, 1, 1]

    def __init__(self, vis, track_id=None):
        """初始化3D人体姿态可视化对象
        
        Args:
            vis: PointcloudVisualizer实例
            track_id: 跟踪ID，用于命名几何体
        """
        self.track_id = track_id
        self.name_prefix = f"track_{track_id}_" if track_id is not None else "single_"
        
        # 创建点云对象
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_name = vis.add_geometry(self.pcd, f"{self.name_prefix}pcd")
        
        # 创建线集对象
        self.lineset = o3d.geometry.LineSet()
        self.lineset_name = vis.add_geometry(self.lineset, f"{self.name_prefix}lineset")
        
        # 创建边界框对象
        self.bbox = o3d.geometry.AxisAlignedBoundingBox()
        self.bbox.color = self.blue
        self.bbox_name = vis.add_geometry(self.bbox, f"{self.name_prefix}bbox")
        
        # 初始化为空状态
        self.update_empty_points()
        self.vis = vis

    def update_empty_points(self):
        """设置为空状态，但保持几何体有效"""
        # 使用一个默认点而不是空点集
        default_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        
        # 更新点云
        self.pcd.points = o3d.utility.Vector3dVector(default_point)
        self.pcd.colors = o3d.utility.Vector3dVector([self.white])
        
        # 更新线集（保持为空但有效）
        self.lineset.points = o3d.utility.Vector3dVector(default_point)
        self.lineset.lines = o3d.utility.Vector2iVector([])
        self.lineset.colors = o3d.utility.Vector3dVector([])
        
        # 更新边界框
        self.bbox.min_bound = default_point[0] - 0.1
        self.bbox.max_bound = default_point[0] + 0.1
        self.bbox.color = self.white

    def update(self, vis):
        """更新所有几何体"""
        vis.update(self.pcd_name)
        vis.update(self.lineset_name)
        vis.update(self.bbox_name)

    def update_data(self, p3d):
        """更新3D姿态数据"""
        # 过滤有效关键点
        valid_index, p3d_colors, p3d_conn, p3d_conn_color = su.filter_valid_kypt_for_o3d_input(p3d)
        
        if len(valid_index) == 0:
            # 如果没有有效关键点，设置为空状态
            self.update_empty_points()
            return
            
        # 更新点云
        self.pcd.points = o3d.utility.Vector3dVector(p3d[valid_index])
        self.pcd.colors = o3d.utility.Vector3dVector(sk_util.kpt_color[valid_index]/255)
        
        # 更新线集
        self.lineset.points = o3d.utility.Vector3dVector(p3d)
        self.lineset.lines = o3d.utility.Vector2iVector(p3d_conn)
        self.lineset.colors = o3d.utility.Vector3dVector(p3d_conn_color)
        
        # 更新边界框
        tempbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox.max_bound = tempbox.max_bound
        self.bbox.min_bound = [tempbox.min_bound[0], tempbox.min_bound[1], 0]  # box Z轴=0
        self.bbox.color = self.blue



class MultiHumanTracker3D:
    """改进的多人3D跟踪器，更好地管理多个3D跟踪对象"""
    
    def __init__(self, vis, max_age=15):
        self.tracks = {}  # 使用track_id作为键
        self.vis = vis
        self.max_age = max_age
        self.last_update = {}  # 记录每个track最后更新的时间
        self.frame_count = 0
        
    def update(self, matches, trackobjs_container, cam_manager, cam_ids):
        """更新3D跟踪器"""
        self.frame_count += 1
        active_tracks = set()
        
        # 处理匹配的跟踪对象
        for match in matches:
            row_idx1, col_idx1, row_idx2, col_idx2, score = match
            track_obj1 = trackobjs_container[row_idx1][col_idx1]
            track_obj2 = trackobjs_container[row_idx2][col_idx2]
            
            # 使用第一个相机的track_id作为唯一标识
            track_id = track_obj1.track.track_id
            active_tracks.add(track_id)
            
            # 三角测量关键点
            hpo3d_i = su.HumanPoseObject_3d([track_obj1.obj, track_obj2.obj])
            hpo3d_i.valid_matched_kypts()
            kypt_pair_i = hpo3d_i.matched_kypts
            
            if len(kypt_pair_i[0]) == 0:
                logging.info(f"跟踪ID {track_id} 没有匹配的关键点")
                continue
                
            p3ds_i = cam_manager.triangulate_points(
                kypt_pair_i[0], kypt_pair_i[1], 
                cam_ids[row_idx1], cam_ids[row_idx2]
            )
            p3ds_coco17_i = su.p3d_2_kypt_17format(hpo3d_i.common_indices, p3ds_i)
            
            # 将 p3ds_coco17_i 中的 None 转换为 np.nan，并确保是 float 类型
            p3ds_coco17_i_np = np.array([[x if x is not None else np.nan for x in p] for p in p3ds_coco17_i], dtype=np.float64)

            # 更新或创建3D跟踪对象
            if track_id in self.tracks:
                self.tracks[track_id].update_data(p3ds_coco17_i_np)
            else:
                self.tracks[track_id] = SingleKypto3d(self.vis, track_id)
                self.tracks[track_id].update_data(p3ds_coco17_i_np)
                logging.info(f"创建新的3D跟踪对象: {track_id}")
        
            # 更新最后活跃时间
            self.last_update[track_id] = self.frame_count
        
        # 移除长时间未更新的跟踪对象
        tracks_to_remove = []
        for track_id in self.tracks:
            if track_id not in active_tracks:
                age = self.frame_count - self.last_update.get(track_id, 0)
                if age > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            # 从可视化中移除
            self.vis.remove_geometry(self.tracks[track_id].pcd_name)
            self.vis.remove_geometry(self.tracks[track_id].lineset_name)
            self.vis.remove_geometry(self.tracks[track_id].bbox_name)
            
            # 从跟踪器中移除
            del self.tracks[track_id]
            del self.last_update[track_id]
            logging.info(f"移除3D跟踪对象: {track_id}")
    
    def update_all(self):
        """更新所有跟踪对象的可视化"""
        for track_id, track in self.tracks.items():
            track.update(self.vis)
    
    def check_fence_intrusion(self, landmark2d_xy):
        """检查是否有跟踪对象侵入围栏区域"""
        intrusion = False
        for track_id, track in self.tracks.items():
            # 获取边界框
            max_bound = np.asarray(track.bbox.max_bound)
            min_bound = np.asarray(track.bbox.min_bound)
            
            # 如果边界框无效，跳过
            if np.all(max_bound == min_bound):
                continue
                
            x_max, y_max, z_max = max_bound
            x_min, y_min, z_min = min_bound
            box_2d = [x_min, y_min, x_max, y_max]
            
            # 计算与围栏的IoU
            iou = su.roi2d(box_2d, landmark2d_xy)
            logging.info(f"iou: {iou}")
            if iou > 0:
                intrusion = True

        
        return intrusion

    def set_all_empty(self):
        """将所有跟踪对象设置为空点"""
        for track_id, track in self.tracks.items():
            track.update_empty_points()
            track.update(self.vis)



class MultiHumanTracker3D_NoO3D:
    """
    多人3D跟踪器，不依赖Open3D。
    管理多个3D跟踪对象，计算边界框，并检查围栏入侵。
    使用 Kalman 滤波器平滑和预测3D边界框。
    """
    
    def __init__(self, max_age=10):
        """
        初始化跟踪器。

        Args:
            max_age (int): 跟踪对象在没有更新的情况下保持活动的最大帧数。
        """
        self.tracks = {}  # Stores {'filter': KalmanFilter, 'keypoints': np.ndarray, 'last_measured_frame': int}
        self.max_age = max_age
        self.last_update = {}  # 记录每个track最后更新的帧计数 (用于 max_age)
        self.frame_count = 0
        
        # Kalman Filter Parameters (tune these based on expected motion and measurement noise)
        # State: [cx, cy, cz, w, h, d] (center, width, height, depth)
        # Measurement: [cx, cy, cz, w, h, d]
        self._dim_x = 6 # Dimension of state vector
        self._dim_z = 6 # Dimension of measurement vector
        # Measurement noise covariance (R): How much noise in bbox calculation from keypoints?
        # Variances for [cx, cy, cz, w, h, d]
        self._R_diag = np.array([0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.1**2], dtype=float) 
        # Process noise covariance (Q): How much random acceleration/change between steps?
        # Assumes relatively smooth motion, adjust if more erratic movement is expected.
        self._Q_diag = np.array([0.05**2, 0.05**2, 0.05**2, 0.01**2, 0.01**2, 0.02**2], dtype=float) 
        # Initial state covariance (P): Initial uncertainty. High value.
        self._P_diag_initial = np.full(self._dim_x, 100.0) 

    def _init_kalman_filter(self, initial_measurement):
        """Initialize a Kalman Filter for a new track."""
        kf = KalmanFilter(dim_x=self._dim_x, dim_z=self._dim_z)
        kf.x = np.array(initial_measurement).reshape(-1, 1) # Initial state = first measurement
        kf.F = np.identity(self._dim_x)       # State transition matrix (constant position model)
        kf.H = np.identity(self._dim_x)       # Measurement function (measure state directly)
        kf.R = np.diag(self._R_diag)          # Measurement noise covariance
        kf.Q = np.diag(self._Q_diag)          # Process noise covariance
        kf.P = np.diag(self._P_diag_initial)  # Initial state covariance
        return kf

    def _calculate_bbox_params(self, keypoints_3d):
        """Calculate bbox parameters [cx, cy, cz, w, h, d] from 3D keypoints."""
        valid_points = keypoints_3d[~np.isnan(keypoints_3d).any(axis=1)]
        if valid_points.shape[0] < 2: # Need at least 2 points for dimensions
            return None 
        
        min_coords = np.min(valid_points, axis=0)
        max_coords = np.max(valid_points, axis=0)
        
        center = (min_coords + max_coords) / 2.0
        dimensions = max_coords - min_coords
        
        # Ensure minimum dimensions to avoid degenerate boxes
        min_dim = 0.01 # 1 cm
        dimensions = np.maximum(dimensions, min_dim) 
        
        return np.concatenate((center, dimensions)) # [cx, cy, cz, w, h, d]

    def _bbox_state_to_list(self, state_vector):
        """Convert Kalman filter state [cx, cy, cz, w, h, d] to AABB list [min_x, ..., max_z]."""
        if state_vector is None or len(state_vector) != self._dim_x:
            return None
        
        center = state_vector[:3].flatten()
        dimensions = state_vector[3:].flatten()
        half_dims = dimensions / 2.0
        
        min_coords = center - half_dims
        max_coords = center + half_dims
        
        return [min_coords[0], min_coords[1], min_coords[2],
                max_coords[0], max_coords[1], max_coords[2]]

    def update(self, matches, trackobjs_container, cam_manager, cam_ids):
        """
        根据跨相机匹配结果更新3D跟踪器。
        Includes Kalman filter prediction and update steps.

        Args:
            matches (list): 跨相机匹配结果列表。
            trackobjs_container (list): 包含每个相机检测到的 TrackHPObj 对象的列表。
            cam_manager (camera.CameraManager): 相机管理器实例。
            cam_ids (list): 相机ID列表。
        """
        self.frame_count += 1
        active_tracks = set()
        
        # 1. Predict step for all existing tracks
        for track_id in self.tracks:
            self.tracks[track_id]['filter'].predict()

        # 2. Update step for matched tracks
        for match in matches:
            row_idx1, col_idx1, row_idx2, col_idx2, score = match
            
            track_obj1 = trackobjs_container[row_idx1][col_idx1]
            track_obj2 = trackobjs_container[row_idx2][col_idx2]
            track_id = track_obj1.track.track_id
            active_tracks.add(track_id)
            
            hpo3d_i = su.HumanPoseObject_3d([track_obj1.obj, track_obj2.obj])
            hpo3d_i.valid_matched_kypts()
            kypt_pair_i = hpo3d_i.matched_kypts
            
            if len(kypt_pair_i[0]) < 3: 
                logging.warning(f"跟踪ID {track_id} 没有足够 ({len(kypt_pair_i[0])}) 的匹配关键点进行三角测量。")
                # Mark as updated so it's not removed immediately if prediction exists
                self.last_update[track_id] = self.frame_count 
                continue
                
            p3ds_i = cam_manager.triangulate_points(
                kypt_pair_i[0], kypt_pair_i[1], 
                cam_ids[row_idx1], cam_ids[row_idx2]
            )
            p3ds_coco17_i = su.p3d_2_kypt_17format(hpo3d_i.common_indices, p3ds_i)
            p3ds_coco17_i_np = np.array([[x if x is not None else np.nan for x in p] for p in p3ds_coco17_i], dtype=np.float64)

            # Calculate measurement for Kalman filter
            measured_bbox_params = self._calculate_bbox_params(p3ds_coco17_i_np)

            if measured_bbox_params is None:
                 logging.warning(f"Track {track_id}: Could not calculate valid bbox from keypoints. Skipping Kalman update.")
                 # Mark as updated so it's not removed immediately if prediction exists
                 self.last_update[track_id] = self.frame_count 
                 continue

            # Update or create 3D tracking object
            if track_id not in self.tracks:
                # Initialize Kalman filter for new track
                kf = self._init_kalman_filter(measured_bbox_params)
                current_bbox = self._bbox_state_to_list(kf.x) # Calculate bbox from initial state
                # 计算扩展的围栏区域
                self.tracks[track_id] = {
                    'filter': kf, 
                    'keypoints': p3ds_coco17_i_np, # Store measured keypoints
                    'bbox': current_bbox,          # Store calculated bbox
                    'last_measured_frame': self.frame_count,
                }
                logging.info(f"创建新的3D跟踪对象并初始化Kalman滤波器: {track_id}")
            else:
                # Update existing filter
                self.tracks[track_id]['filter'].update(measured_bbox_params)
                updated_state = self.tracks[track_id]['filter'].x
                current_bbox = self._bbox_state_to_list(updated_state) # Calculate bbox from updated state
                # Store the latest measured keypoints and updated bbox
                self.tracks[track_id]['keypoints'] = p3ds_coco17_i_np 
                self.tracks[track_id]['bbox'] = current_bbox
                self.tracks[track_id]['last_measured_frame'] = self.frame_count
            # Update timestamp for max_age calculation
            self.last_update[track_id] = self.frame_count
        
        # 3. Clean up stale tracks
        tracks_to_remove = []
        for track_id in list(self.tracks.keys()): 
            # Use last_update which is updated even if only prediction happened implicitly before matches
            age = self.frame_count - self.last_update.get(track_id, self.frame_count) 
            logging.info(f"track_id: {track_id}, age: {age}")
            if age > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.last_update:
                del self.last_update[track_id]
            logging.info(f"++++++++++移除过时的3D跟踪对象 (超过 max_age), track_id = {track_id}")

    def check_fence_intrusion(self, landmark2d_xy, expand_ratio=0.1):
        """
        检查是否有跟踪对象的地面投影边界框（来自Kalman滤波器）侵入指定的2D围栏区域。

        Args:
            landmark2d_xy (list or np.array): 定义围栏区域的2D边界 [x_min, y_min, x_max, y_max]。

        Returns:
            bool: 如果有任何跟踪对象侵入围栏，则返回 True，否则返回 False。
        """
        intrusion = False
        for track_id, track_data in self.tracks.items():
            # Get the current (potentially predicted) state from the filter
            current_state = track_data['filter'].x
            bbox = self._bbox_state_to_list(current_state)

            if bbox is None:
                logging.debug(f"Track {track_id} has no valid bbox state for intrusion check.")
                continue
                
            x_min, y_min, _, x_max, y_max, _ = bbox
            # 计算扩展的围栏区域    
            width = x_max - x_min
            height = y_max - y_min
            expanded_bbox = [x_min - width * expand_ratio, y_min - height * expand_ratio,
                            x_max + width * expand_ratio, y_max + height * expand_ratio]
            box_2d_projection = expanded_bbox
            
            iou = su.roi2d(box_2d_projection, landmark2d_xy)
            
            if iou > 0.01: 
                logging.info(f"入侵检测: Track {track_id} (Filtered BBox IoU: {iou:.3f})")
                intrusion = True
                # break # Uncomment if only one intrusion needs to be detected
        
        return intrusion

    def set_all_empty(self):
        """
        将所有跟踪对象的关键点数据设置为空（NaN）。
        Kalman 滤波器将继续基于最后一次有效测量进行预测。
        """
        for track_id, track_data in self.tracks.items():
            if 'keypoints' in track_data:
                 track_data['keypoints'] = np.full((17, 3), np.nan)
                 # Do not reset the filter, let it predict
                 # Mark measurement as old, prediction will rely on this flag implicitly if needed
                 track_data['last_measured_frame'] = -1 # Or some indicator of staleness
            # No visualization update needed

    def get_tracked_points(self):
        """
        获取所有当前跟踪对象的最后测量的3D关键点。
        注意：如果 track 的 last_measured_frame != self.frame_count，则这些点是旧的。

        Returns:
            dict: 以 track_id 为键，3D关键点 (np.ndarray(17,3)) 为值的字典。
        """
        return {track_id: data.get('keypoints') for track_id, data in self.tracks.items()}

    def get_tracked_bboxes(self):
        """
        获取所有当前跟踪对象的当前估计的3D边界框（来自Kalman滤波器）。
        这可能是基于当前帧的更新或预测。

        Returns:
            dict: 以 track_id 为键，边界框 ([min_x, ..., max_z]) 为值的字典。
        """
        bboxes = {}
        for track_id, data in self.tracks.items():
            current_state = data['filter'].x
            bboxes[track_id] = self._bbox_state_to_list(current_state)
        return bboxes

    



