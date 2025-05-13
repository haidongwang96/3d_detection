import os
import numpy as np

import cv2
import glob
import camera
import utility as su

green = (0,255,0)
red = (0,0,255)
blue = (255,0,0)
yellow = (0,255,255)
black = (0,0,0)
font_scale = 1.5
object_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0,1,0]], dtype=np.float32)
z_axis = np.array((0,0,1), dtype=np.float32)

def on_EVENT_LBUTTONDOWN(event, x, y, args, param):
    img, click_index, click_coord = param
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(x,y)
        click_coord.append([x,y])
        cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=-1)
        cv2.putText(img, str(click_index[0]), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 255, 0), thickness=1)
        cv2.imshow("image", img)
        click_index[0] += 1

def single_landmark_annotation(proj_folder, image_folder_path, cam_id):


    images_names = su.collect_images_by_index(image_folder_path, cam_id)
    assert len(images_names) == 1, ("外参为唯一图像")
    landmark_image_path = images_names[0]

    img = cv2.imread(landmark_image_path)
    img_coord = img.copy()
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    click_index = [0]
    click_coord = []
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, (img, click_index, click_coord))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

    assert len(click_coord) == 4, ("只支持4边型")
    img = su.draw_fence(img, click_coord, color=(0,0,255))
    cv2.imwrite(landmark_image_path.replace(".jpg","_draw.jpg"),img)
    su.write_list_file(click_coord, landmark_image_path.replace(".jpg", ".txt"))


    intr_path = os.path.join(proj_folder,"calib_info",f"camera_intrinsic_cam_{cam_id}.json")
    intr = camera.intr_load(intr_path)
    K = intr.get_cam_mtx()
    d = intr.dist

    p_oxy = np.array(click_coord,dtype=np.float32)
    p_o = click_coord[0]
    p_x = click_coord[1]
    p_y = click_coord[3]

    # solvePnP 返回的旋转是从世界坐标系到相机坐标系的旋转。
    success, rvec, tvec = cv2.solvePnP(object_points, p_oxy, K, d)  # 使用AP3P算法处理3个点)
    if not success:
        raise ValueError("solvePnP failed to compute a solution.")

    pz2d, _ = cv2.projectPoints(z_axis, rvec, tvec, K, d)
    p_z = pz2d[0][0].astype(int)

    cv2.circle(img_coord, (int(p_o[0]), int(p_o[1])), 2, green, 2)
    cv2.circle(img_coord, (int(p_x[0]), int(p_x[1])), 2, red, 2)
    cv2.circle(img_coord, (int(p_y[0]), int(p_y[1])), 2, blue, 2)

    cv2.line(img_coord, p_o, p_x, red, 1)
    cv2.putText(img_coord,"x-axis",p_x,cv2.FONT_HERSHEY_PLAIN,font_scale,black,2)
    cv2.line(img_coord, p_o, p_y, blue, 1)
    cv2.putText(img_coord, "y-axis", p_y, cv2.FONT_HERSHEY_PLAIN, font_scale, black,2)

    cv2.circle(img_coord, (int(p_z[0]), int(p_z[1])), 2, green, 2)
    cv2.line(img_coord, p_o, (int(p_z[0]), int(p_z[1])), yellow, 1)
    cv2.putText(img_coord, "z-axis", p_z, cv2.FONT_HERSHEY_PLAIN, font_scale, black,2)

    cv2.imshow(f"show", img_coord)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(landmark_image_path.replace(".jpg","_axis.jpg"),img_coord)

    extr = camera.Extrinsic(rvec, tvec)
    save_dir = os.path.dirname(intr_path)
    extr_path_aruco_c = os.path.join(save_dir, f"camera_extrinsic_landmark_{cam_id}.json")
    extr.save(extr_path_aruco_c)



def multi_landmark_annotation(landmark_dir, cam_ids=[0, 1]):
    for cam_id in cam_ids:
        single_landmark_annotation(landmark_dir, cam_id)




if __name__ == '__main__':


    cam_ids = [2,4]
    print("点出四个点，第一个点为原点0,(0,0,0)，第二个点为x轴方向（1,0,0），第三个点（1,1,0）,第四个点为y轴方向（0,1,0")
    landmark_folder = "../data/record/office_3rd_floor_whd/landmark_0"
    multi_landmark_annotation(landmark_folder, cam_ids)






