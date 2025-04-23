
import sys
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径

import utility as su

import cv2
import numpy as np

import camera


def compute_extrinsic_matrix(object_points, img_points, K):

    success, rvec, tvec = cv2.solvePnP(object_points, img_points, K, None)  # 使用AP3P算法处理3个点)
    if not success:
        raise ValueError("solvePnP failed to compute a solution.")
    return rvec, tvec



config = su.read_yaml_file('../data/record/office_3rd_floor_whd/config.yaml')
CAM_IDS = config["cam_ids"]
multi_cam = camera.MultiCameraCapture(config)

cam0 = multi_cam.Cameras[0]
cam1 = multi_cam.Cameras[1]
k0 = cam0.intr.get_cam_mtx()
k1 = cam1.intr.get_cam_mtx()

landmark0_pic_path = "../data/record/office_3rd_floor_whd/landmark_0/1732607373680_2.jpg"
landmark1_pic_path = "../data/record/office_3rd_floor_whd/landmark_0/1732607373680_4.jpg"

landmark0_txt_path = landmark0_pic_path.replace("jpg","txt")
landmark1_txt_path = landmark1_pic_path.replace("jpg","txt")

img0 = cv2.imread(landmark0_pic_path)
img1 = cv2.imread(landmark1_pic_path)

mark0 = su.read_list_file(landmark0_txt_path," ")
mark1 = su.read_list_file(landmark1_txt_path," ")

imgs = [img0, img1]
marks = [mark0, mark1]
ks = [k0, k1]

green = (0,255,0)
red = (0,0,255)
blue = (255,0,0)
yellow = (0,255,255)
black = (0,0,0)
font_scale = 1.5

object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1,1,0]], dtype=np.float32)
z_axis = np.array((0,0,1), dtype=np.float32)
x_min = 0
x_max = 1
y_min = 0
y_max = 1
z_min = 0
z_max = 1

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



def solve_XY(u, v, k, R, t):
    # 构造齐次坐标
    uv_homogeneous = np.array([u, v, 1])

    # 计算投影矩阵 P = k[R|t]
    P = np.dot(k, np.hstack((R[:, :2], t.reshape(3, 1))))

    # 计算逆矩阵
    P_inv = np.linalg.pinv(P)

    # 求解 [X, Y, 1]
    XY_homogeneous = np.dot(P_inv, uv_homogeneous)

    # 归一化
    X = XY_homogeneous[0] / XY_homogeneous[2]
    Y = XY_homogeneous[1] / XY_homogeneous[2]

    return X, Y

extrs= []
for i,(img, mark) in enumerate(zip(imgs, marks)) :

    mark = np.array(mark, dtype=int)
    p_o = mark[0]
    p_x = mark[1]
    p_y = mark[3]
    p_xy = mark[2]

    p_oxy = np.array([p_o, p_x, p_y, p_xy],dtype=np.float32)

    rvec, tvec = compute_extrinsic_matrix(object_points, p_oxy, ks[i])
    #extrs.append(extr)

    cv2.circle(img, (int(p_o[0]), int(p_o[1])), 2, green, 2)
    cv2.circle(img, (int(p_x[0]), int(p_x[1])), 2, red, 2)
    cv2.circle(img, (int(p_y[0]), int(p_y[1])), 2, blue, 2)

    cv2.line(img, p_o, p_x, red, 1)
    cv2.putText(img,"x-axis",p_x,cv2.FONT_HERSHEY_PLAIN,font_scale,black,2)
    cv2.line(img, p_o, p_y, blue, 1)
    cv2.putText(img, "y-axis", p_y, cv2.FONT_HERSHEY_PLAIN, font_scale, black,2)

    p2d, _ = cv2.projectPoints(z_axis, rvec, tvec, ks[i], None)
    p_z = p2d[0][0].astype(int)

    cv2.circle(img, (int(p_z[0]), int(p_z[1])), 2, green, 2)
    cv2.line(img, p_o, (int(p_z[0]), int(p_z[1])), yellow, 1)
    cv2.putText(img, "z-axis", p_z, cv2.FONT_HERSHEY_PLAIN, font_scale, black,2)

    p_test = np.array([0.5, 0.5, 0])
    p2_test, _ = cv2.projectPoints(p_test, rvec, tvec, ks[i], None)
    p2_test = p2_test[0][0].astype(int)
    print(p2_test)

    cv2.circle(img, (int(p2_test[0]), int(p2_test[1])), 2, green, 2)
    #cv2.line(img, p_o, (int(p_z[0]), int(p_z[1])), yellow, 1)
    #cv2.putText(img, "z-axis", p_z, cv2.FONT_HERSHEY_PLAIN, font_scale, black,2)

    extr = camera.Extrinsic(rvec, tvec)
    R = extr.R()
    t = extr.t()

    x,y = solve_XY(p2_test[0],p2_test[1], ks[i], R, t)
    print(x,y,0)

    vertices_2d =  project_8_vertices_to_imgpoint(x_min, x_max, y_min, y_max, z_min, z_max, ks[i], rvec, tvec)

    for edge in edges:
        pt1 = tuple(vertices_2d[edge[0]])
        pt2 = tuple(vertices_2d[edge[1]])
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)



stack = np.hstack((img0, img1))
cv2.imshow(f"Camera", stack)
# 按下 'q' 键退出
key = cv2.waitKey(0)
cv2.destroyAllWindows()