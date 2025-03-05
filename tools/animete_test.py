import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3D


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import camera
import utility as su

p3ds_all = su.read_pickle_file("../p3d.pkl")

landmark3ds = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
boundary = [(-1, 2), (-1, 2), (0, 2)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


sk_util = su.skeleton_util()


scat_colors = sk_util.kpt_color
# 初始化可视化元素
scat = ax.scatter([], [], [], c='blue', s=50)  # 关节点

#lines = [ax.add_line(Line3D([], [], [], c='r', lw=2)) for _ in sk_util.skeleton]  # 骨架线
ax.set_xlim(boundary[0][0], boundary[0][1])
ax.set_ylim(boundary[1][0], boundary[1][1])
ax.set_zlim(boundary[2][0], boundary[2][1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# # 初始化函数，设定初始状态
# def init():
#     """初始化元素数据"""
#     scat._offsets3d = ([], [], [])
#     for line in lines:
#         line.set_data_3d([], [], [])
#     return [scat] + lines

# 更新函数，用于每一帧的更新
def update(frame):

    """更新每帧数据"""
    # 获取当前帧数据
    p3ds = p3ds_all[frame]


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

    # 更新关节点位置
    scat._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])

    # 更新骨架连线
    for i, (start, end) in enumerate(skeleton_connections):
        # 获取当前连接线的起点和终点坐标
        x = [frame_data[start, 0], frame_data[end, 0]]
        y = [frame_data[start, 1], frame_data[end, 1]]
        z = [frame_data[start, 2], frame_data[end, 2]]

        # 更新线条数据
        lines[i].set_data_3d(x, y, z)

    return [scat] + lines


# 创建动画
ani = animation.FuncAnimation(
    fig, update, frames=len(p3ds), init_func=init, blit=True, interval=50
)

# 显示动画
plt.show()


