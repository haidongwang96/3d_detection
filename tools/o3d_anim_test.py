import cv2
import os
import time
import numpy as np
import open3d as o3d


from camera import ProjectionMatrix

print(o3d.__version__)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import camera
import utility as su

sk_util = su.skeleton_util()
class PointcloudVisualizer():

    def __init__(self, camera_params):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.camera_params = camera_params
        #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #self.vis.add_geometry(mesh)


        self.ctr = self.vis.get_view_control()


        self.Initialized = False
        print("init")

    # self.vis.register_key_callback(key, your_update_function)

    def add_geometry(self, cloud):
        self.vis.add_geometry(cloud)

    def remove_geometry(self, cloud):
        self.vis.remove_geometry(cloud)

    def update(self, cloud):
        # Your update routine
        # 应用相机参数到视图
        self.ctr.convert_from_pinhole_camera_parameters(self.camera_params, allow_arbitrary=True)
        self.vis.update_geometry(cloud)
        self.vis.update_renderer()
        self.vis.poll_events()

    def destroy(self):
        self.vis.destroy_window()



def filter_valid_conn(p3d):
    # p3ds : (17,3)
    p3d_indices = np.where((p3d[:, 0] != 0) & (p3d[:, 1] != 0) & (p3d[:, 2] != 0))[0]
    p3d_conn = []
    p3d_conn_color = []
    for i, sk in enumerate(sk_util.skeleton):
        pos1 = int(sk[0]-1)
        pos2 = int(sk[1]-1)
        if pos1 in p3d_indices and pos2 in p3d_indices:
            p3d_conn.append([sk[0]-1, sk[1]-1])
            p3d_conn_color.append(sk_util.limb_color[i]/255)

    return p3d_indices, p3d_conn, p3d_conn_color

def main():
    config = su.read_yaml_file('../data/record/office_3rd_floor_whd/config.yaml')

    multi_cam = camera.MultiCameraCapture(config)
    intr_mat = multi_cam.Cameras[0].intr.get_cam_mtx()
    extr_mat = multi_cam.Cameras[0].extr.pose_mat

    intrinsic = o3d.camera.PinholeCameraIntrinsic(500,300,intr_mat)
    #intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intr_mat)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extr_mat


    data = su.read_pickle_file("../p3d.pkl")


    landmark3ds = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    landmark_conn = [[0,1],[1,2],[2,3],[3,0]]
    landmark_color = [1,0,0]
    boundary = [(-1, 2), (-1, 2), (0, 2)]


    # 初始化可视化窗口
    vis = PointcloudVisualizer(camera_params)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(list([[0.0, 0.0, 0.0]]))
    lineset.lines = o3d.utility.Vector2iVector([[0,1]])
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (0, 1, 0)


    landmark_lineset = o3d.geometry.LineSet()
    landmark_lineset.points = o3d.utility.Vector3dVector(landmark3ds)
    landmark_lineset.lines = o3d.utility.Vector2iVector(landmark_conn)
    landmark_lineset.colors = o3d.utility.Vector3dVector([landmark_color for _ in landmark_conn])  # 红色线条)

    vis.add_geometry(pcd)
    vis.add_geometry(lineset)
    vis.add_geometry(bbox)
    vis.add_geometry(landmark_lineset)


    # 逐帧更新动画
    for i, p3d in enumerate(data):
        # 更新LineSet和PointCloud的坐标

        valid_index, p3d_conn,p3d_conn_color = filter_valid_conn(p3d)

        pcd.points = o3d.utility.Vector3dVector(p3d[valid_index])
        pcd.colors = o3d.utility.Vector3dVector(sk_util.kpt_color[valid_index]/255)

        lineset.points = o3d.utility.Vector3dVector(p3d)
        lineset.lines = o3d.utility.Vector2iVector(p3d_conn)
        lineset.colors = o3d.utility.Vector3dVector(p3d_conn_color)

        tempbox = pcd.get_axis_aligned_bounding_box()
        bbox.max_bound = tempbox.max_bound
        bbox.min_bound = tempbox.min_bound

        vis.update(pcd)
        vis.update(lineset)
        vis.update(bbox)

        time.sleep(0.05)

    vis.destroy()

if __name__ == '__main__':
    main()