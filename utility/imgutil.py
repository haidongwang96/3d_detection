import cv2
import utility as su
import numpy as np

fontScale = 0.5
thickness = 1


edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 侧面
]

def project_8_vertices_to_imgpoint(x_min, x_max, y_min, y_max, z_min, z_max, k, rvec,tvec):
    vertices_3d = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ]).astype(float)

    p2d, _ = cv2.projectPoints(vertices_3d, rvec, tvec, k, None)
    p2d = p2d.astype(int).reshape((len(p2d), 2))

    return p2d


def project_keypoints_to_imgpoint(keypoints, k, rvec, tvec):
    p2d, _ = cv2.projectPoints(keypoints, rvec, tvec, k, None)
    p2d = p2d.astype(int).reshape((len(p2d), 2))
    return p2d


def get_axis_points(length=0.05):
    coordinate_points = np.float32([[0, 0, 0],  # 原点
                                    [length, 0, 0],  # X轴终点
                                    [0, length, 0],  # Y轴终点
                                    [0, 0, length],  # Z轴终点
                                ]).reshape(-1, 3)
    return coordinate_points

def draw_axes(frame, imgpts):
    assert len(imgpts) == 4
    cv2.line(frame, imgpts[0], imgpts[1], (0, 0, 255), 3)
    cv2.line(frame, imgpts[0], imgpts[2], (0, 255, 0), 3)
    cv2.line(frame, imgpts[0], imgpts[3], (255, 0, 0), 3)
    return frame




def drawCube(img, imgpts, color=(0, 255, 0), thickness=1):
    """ Draws a 3D cube projected onto a 2D image plane.

    Args:
        img: The image to draw on.
        imgpts: An array of 8 projected 2D points corresponding to the cube vertices.
                Expected order: (bottom face counter-clockwise, then top face counter-clockwise)
                0: x_min, y_min, z_min -> projected
                1: x_max, y_min, z_min -> projected
                2: x_max, y_max, z_min -> projected
                3: x_min, y_max, z_min -> projected
                4: x_min, y_min, z_max -> projected
                5: x_max, y_min, z_max -> projected
                6: x_max, y_max, z_max -> projected
                7: x_min, y_max, z_max -> projected
        color: The color of the cube lines (B, G, R).
        thickness: The thickness of the cube lines.
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Define the 12 edges of the cube based on vertex indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Draw each edge
    for pt1_idx, pt2_idx in edges:
        pt1 = tuple(imgpts[pt1_idx])
        pt2 = tuple(imgpts[pt2_idx])
        img = cv2.line(img, pt1, pt2, color, thickness)

    return img

def draw_o3d_cube(img, pts, color):
    pts = np.int32(pts)

    for i,j in [[0,3],[1,6],[2,5],[7,4],[3,6],[6,4],[4,5],[5,3],[0,2],[2,7],[7,1],[1,0]]:
        img = cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 3)
    return  img



def draw_predict(region, image, color=(0, 255, 0), bbox=True, polyline=False):


    if polyline and region.polygon is not None:

        cv2.polylines(image, [region.polygon.astype('int')], True, color, thickness)
        #cv2.fillPoly(image, [region.polygon.astype('int')], color, cv2.LINE_AA)

    if bbox:
        x1, y1, x2, y2 = region.toint().bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{region.label} {region.score:.2f} ", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, color, thickness, cv2.LINE_AA)

    return image


def generate_distinct_colors(count):
    """生成count个不同的颜色.

    方法来自: http://blog.csdn.net/yhl_leo/article/details/52185581
    """

    assert count < 256, "this method can only generate 255 different colors."

    colors = []
    for i in range(count):
        r, g, b, ii = (0, 0, 0, i)
        for j in range(7):
            str_ii = f"{bin(ii)[2:]:0>8}"[-8:]
            r = r ^ (int(str_ii[-1]) << (7 - j))
            g = g ^ (int(str_ii[-2]) << (7 - j))
            b = b ^ (int(str_ii[-3]) << (7 - j))
            ii = ii >> 3
        colors.append((b, g, r))
    return colors



def draw_track_bbox(frame, bbox, id, color, thickness=1):

    x1, y1, x2, y2 = bbox
    cv2.putText(frame, f"TrackID: {id}", (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return frame

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
class EasyAnnotator:

    def __init__(self, im, lw=2):
        self.im = im.copy()
        self.lw = lw

        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def text_label(self, box, label, txt_color=(0, 255, 0), thickness=2):
        """
        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            txt_color (tuple, optional): The color of the text (R, G, B).
        """
        x1, y1, x2, y2 = box

        cv2.putText(
            self.im,
            label,
            (int(x1-5), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            txt_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_box(self, box, color=(0, 255, 0), thickness=2):
        """
        box (tuple): The bounding box coordinates (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(self.im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    def kpts(self, kpts, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
        """
        Plot keypoints on the image.

        Args:
            kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
            shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
            radius (int, optional): Keypoint radius. Defaults to 5.
            kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.

        Note:
            - `kpt_line=True` currently only supports human pose plotting.
            - Modifies self.im in-place.
        """
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = kpt_color or (self.kpt_color[i].tolist() if is_pose else colors(i))
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < conf_thres or conf2 < conf_thres:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(
                        self.im,
                        pos1,
                        pos2,
                        kpt_color or self.limb_color[i].tolist(),
                        thickness=int(np.ceil(self.lw / 2)),
                        lineType=cv2.LINE_AA,
                    )



