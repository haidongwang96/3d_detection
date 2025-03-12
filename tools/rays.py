import numpy as np

def closest_distance_between_rays(K1, R1, t1, u1, v1, K2, R2, t2, u2, v2):
    """
    计算两个相机投影射线在空间中的最近距离
    :param K1: 相机1的内参矩阵 (3x3)
    :param R1: 相机1的旋转矩阵 (3x3)
    :param t1: 相机1的平移向量 (3x1)
    :param u1, v1: 相机1的图像点坐标
    :param K2: 相机2的内参矩阵 (3x3)
    :param R2: 相机2的旋转矩阵 (3x3)
    :param t2: 相机2的平移向量 (3x1)
    :param u2, v2: 相机2的图像点坐标
    :return: 两条射线之间的最近距离
    """
    # 将图像点转换为齐次坐标
    point1 = np.array([u1, v1, 1])
    point2 = np.array([u2, v2, 1])

    # 将图像点转换到相机坐标系
    x1 = np.linalg.inv(K1) @ point1
    x2 = np.linalg.inv(K2) @ point2

    # 将相机坐标系中的点转换到世界坐标系
    P1 = np.linalg.inv(R1) @ (x1 - t1)
    P2 = np.linalg.inv(R2) @ (x2 - t2)

    # 相机中心（世界坐标系）
    C1 = -np.linalg.inv(R1) @ t1
    C2 = -np.linalg.inv(R2) @ t2

    # 方向向量（世界坐标系）
    D1 = P1 - C1
    D2 = P2 - C2

    # 归一化方向向量
    D1 = D1 / np.linalg.norm(D1)
    D2 = D2 / np.linalg.norm(D2)

    # 计算两条射线的最近距离
    cross_D1_D2 = np.cross(D1, D2)
    norm_cross = np.linalg.norm(cross_D1_D2)

    if norm_cross < 1e-6:  # 如果射线平行
        distance = np.linalg.norm(np.cross(C2 - C1, D1)) / np.linalg.norm(D1)
    else:  # 如果射线不平行
        distance = np.abs(np.dot(C2 - C1, cross_D1_D2)) / norm_cross

    return distance


# 示例数据
K1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # 相机1内参
R1 = np.eye(3)  # 相机1旋转矩阵（单位矩阵）
t1 = np.array([[0], [0], [0]])  # 相机1平移向量
u1, v1 = 320, 240  # 相机1图像点

K2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # 相机2内参
R2 = np.eye(3)  # 相机2旋转矩阵（单位矩阵）
t2 = np.array([[1], [0], [0]])  # 相机2平移向量
u2, v2 = 320, 240  # 相机2图像点

# 计算最近距离
distance = closest_distance_between_rays(K1, R1, t1, u1, v1, K2, R2, t2, u2, v2)
print(f"两条射线之间的最近距离: {distance}")