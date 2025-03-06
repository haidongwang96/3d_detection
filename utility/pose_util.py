import os

import cv2
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            kpts_3d.append([0, 0, 0])
        kpts_3d.append(p3d)

    while len(kpts_3d) < 17:
        kpts_3d.append([0, 0, 0])

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

    def __init__(self, camera_params):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.camera_params = camera_params
        #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #self.vis.add_geometry(mesh)

        self.ctr = self.vis.get_view_control()

    # self.vis.register_key_callback(key, your_update_function)

    def add_geometry(self, cloud):
        self.vis.add_geometry(cloud)

    def remove_geometry(self, cloud):
        self.vis.remove_geometry(cloud)

    def update(self, cloud):
        # todo: 优化
        # 应用相机参数到视图
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params, allow_arbitrary=True)
        self.vis.update_geometry(cloud)
        self.vis.update_renderer()
        self.vis.poll_events()

    def destroy(self):
        self.vis.destroy_window()


class SingleKypto3d:

    def __init__(self, vis):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))

        self.lineset = o3d.geometry.LineSet()
        self.lineset.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))
        self.lineset.lines = o3d.utility.Vector2iVector([[0, 1]])

        self.bbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox.color = (0, 0, 1)

        vis.add_geometry(self.pcd)
        vis.add_geometry(self.lineset)
        vis.add_geometry(self.bbox)


    def update_empty_points(self):
        self.pcd.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))
        self.lineset.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))
        self.lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
        tempbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox.max_bound = tempbox.max_bound
        self.bbox.min_bound = tempbox.min_bound


    def update(self, vis):
        vis.update(self.pcd)
        vis.update(self.lineset)
        vis.update(self.bbox)

    def update_data(self, p3d):

        valid_index, p3d_colors, p3d_conn, p3d_conn_color = su.filter_valid_kypt_for_o3d_input(p3d)

        self.pcd.points = o3d.utility.Vector3dVector(p3d[valid_index])
        self.pcd.colors = o3d.utility.Vector3dVector(sk_util.kpt_color[valid_index]/255)

        self.lineset.points = o3d.utility.Vector3dVector(p3d)
        self.lineset.lines = o3d.utility.Vector2iVector(p3d_conn)
        self.lineset.colors = o3d.utility.Vector3dVector(p3d_conn_color)

        tempbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox.max_bound = tempbox.max_bound
        self.bbox.min_bound = tempbox.min_bound



class PoseAnimater:

    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)

    def pose_animation(self, p3ds):

        ani = animation.FuncAnimation(
            self.fig,
            self.update_pose,
            frames=100,
            init_func=self.ani_init(),
            blit=True,
            interval=50,
            fargs=(p3ds,)
)


    def ani_init(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')



    def update_pose(self, p3ds):
        for i, k in enumerate(p3ds):
            x, y, z = k
            if x == y == z == 0:
                continue
            self.ax.scatter(x, y, z, color=sk_util.kpt_color[i] / 255, marker='o')

        for i, sk in enumerate(sk_util.skeleton):
            pos0 = (p3ds[(sk[0] - 1), 0], p3ds[(sk[0] - 1), 1], p3ds[(sk[0] - 1), 2])
            pos1 = (p3ds[(sk[1] - 1), 0], p3ds[(sk[1] - 1), 1], p3ds[(sk[1] - 1), 2])
            if pos0[0] == pos0[1] == pos0[2] == 0 or pos1[0] == pos1[1] == pos1[2] == 0:
                continue
            self.ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], [pos0[2], pos1[2]],
                    color=sk_util.limb_color[i] / 255, linestyle='-')

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.pause(0.01)  # 暂停一小段时间以模拟逐帧更新



