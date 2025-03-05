import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

# ===== 配置参数 =====
num_frames = 100  # 总帧数
n_joints = 17  # 关节点数量

# 生成示例数据 (num_frames, 17, 3)
data = np.random.rand(num_frames, n_joints, 3) * 10

# 定义骨架连接关系 (COCO格式示例)
skeleton_connections = [
    [15, 13], [13, 11], [16, 14], [14, 12],  # 四肢
    [11, 12], [5, 11], [6, 12],  # 躯干
    [5, 7], [7, 9], [6, 8], [8, 10],  # 手臂
    [0, 1], [1, 2], [0, 3], [3, 4]  # 头部
]

# ===== 创建图形 =====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始化可视化元素
scat = ax.scatter([], [], [], c='blue', s=50)  # 关节点
lines = [ax.add_line(Line3D([], [], [], c='r', lw=2)) for _ in skeleton_connections]  # 骨架线

# 设置坐标轴动态范围
ax.set_xlim(data[..., 0].min() - 1, data[..., 0].max() + 1)
ax.set_ylim(data[..., 1].min() - 1, data[..., 1].max() + 1)
ax.set_zlim(data[..., 2].min() - 1, data[..., 2].max() + 1)


# ===== 动画函数 =====
def init():
    """初始化元素数据"""
    scat._offsets3d = ([], [], [])
    for line in lines:
        line.set_data_3d([], [], [])
    return [scat] + lines


def update(frame):
    """更新每帧数据"""
    # 获取当前帧数据
    frame_data = data[frame]

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


# ===== 创建动画 =====
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    init_func=init,
    blit=True,
    interval=50,
    repeat=False
)

plt.show()