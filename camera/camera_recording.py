import numpy as np
import cv2
import time
import os

import utility as su


def get_cv2_capture(cam_id, config=None):
    """
    此函数的目的是同一所有opencv的视频流设定
    Note: 采用不同分辨率的相机，内参不同
    """
    if config is not None:
        width = config['width']
        height = config['height']
        fps = config['fps']
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)  # 设置帧率为30fps
    else:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为30fps

    return cap

def single_picture_recording(cam_id,key="sample"):
    sample_folder = su.create_ouput_folder(key)

    print("single camera recording")
    cap = get_cv2_capture(cam_id)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("0", frame)
        key = cv2.waitKey(1)
        if key == 32:
            timestampSTR = str(round(time.time() * 1000))

            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"saved Image:  {img_path}")

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("0")

def double_picture_recording(cam_id1, cam_id2, key="sample"):

    sample_folder = su.create_ouput_folder(key)
    print("double camera recording")
    # capture0 = cv2.VideoCapture(cam_id1)
    # capture1 = cv2.VideoCapture(cam_id2)
    capture0 = get_cv2_capture(cam_id1)
    capture1 = get_cv2_capture(cam_id2)

    if not capture0.isOpened() or not capture1.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    window_name = f"{cam_id1} + {cam_id2}"

    while True:
        t0 = time.time()
        # Capture frames from both cameras
        ret0, frame0 = capture0.read()
        ret1, frame1 = capture1.read()

        # Check if frames were captured successfully
        if not ret0 or not ret1:
            print("Error: Could not read frames from one or both cameras.")
            break

        images = np.hstack((frame0, frame1))
        cv2.imshow(window_name, images)
        key = cv2.waitKey(1)

        if key == 32:
            timestampSTR = str(round(time.time() * 1000))
            # for stream0
            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id1}.jpg")
            cv2.imwrite(img_path, frame0)
            print(f"saved Image:  {img_path}")

            # for stream1
            img_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id2}.jpg")
            cv2.imwrite(img_path, frame1)
            print(f"saved Image:  {img_path}")

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


def double_mp4_recording(cam_id1, cam_id2, key="video", config =None):


    fps = 30
    width= 1280
    height = 720

    if config is not None:
        width = config['width']
        height = config['height']
        fps = config['fps']

    sample_folder = su.create_ouput_folder(key)
    print("double camera video recording")

    capture0 = get_cv2_capture(cam_id1)
    capture1 = get_cv2_capture(cam_id2)

    timestampSTR = str(round(time.time() * 1000))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式编码

    cap0_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id1}.mp4")
    cap1_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id2}.mp4")
    out0 = cv2.VideoWriter(cap0_path, fourcc, fps, (width, height)) # 这里需要和get_cv2_capture （height,width)对齐
    out1 = cv2.VideoWriter(cap1_path, fourcc, fps, (width, height))


    recording = False

    if not capture0.isOpened() or not capture1.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    while True:
        ret0, frame0 = capture0.read()
        ret1, frame1 = capture1.read()

        if not ret0 or not ret1:
            print("Error: Could not read frames from one or both cameras.")
            break

        images = np.hstack((frame0, frame1))
        cv2.imshow("0", images)
        key = cv2.waitKey(1)

        if key == 32:
            recording = True
            print("start recording")

        if recording:
            out0.write(frame0)
            out1.write(frame1)

        if key == 27:  # exit on ESC
            break

    capture0.release()
    out0.release()
    capture1.release()
    out1.release()
    cv2.destroyWindow("0")


def triple_double_mp4_recording(cam_id0, cam_id1, cam_id2, key="video", config=None):


    fps = 30
    width= 1280
    height = 720

    if config is not None:
        width = config['width']
        height = config['height']
        fps = config['fps']

    sample_folder = su.create_ouput_folder(key)
    print("triple camera video recording")

    capture0 = get_cv2_capture(cam_id0)
    capture1 = get_cv2_capture(cam_id1)
    capture2 = get_cv2_capture(cam_id2) 

    timestampSTR = str(round(time.time() * 1000))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式编码

    cap0_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id0}.mp4")
    cap1_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id1}.mp4")
    cap2_path = os.path.join(sample_folder, f"{timestampSTR}_{cam_id2}.mp4")
    out0 = cv2.VideoWriter(cap0_path, fourcc, fps, (width, height)) # 这里需要和get_cv2_capture （height,width)对齐
    out1 = cv2.VideoWriter(cap1_path, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(cap2_path, fourcc, fps, (width, height))

    recording = False

    if not capture0.isOpened() or not capture1.isOpened():
        print("Error: Could not open one or both cameras.")
        exit()

    while True:
        ret0, frame0 = capture0.read()
        ret1, frame1 = capture1.read()
        ret2, frame2 = capture1.read()

        if not ret0 or not ret1 or not ret2:
            print("Error: Could not read frames from 1/2/3 cameras.")
            break

        images = np.hstack((frame0, frame1,frame2))
        cv2.imshow("0", images)
        key = cv2.waitKey(1)

        if key == 32:
            recording = True
            print("start recording")

        if recording:
            out0.write(frame0)
            out1.write(frame1)
            out2.write(frame2)

        if key == 27:  # exit on ESC
            break

    capture0.release()
    out0.release()
    capture1.release()
    out1.release()
    capture2.release()
    out2.release()
    cv2.destroyWindow("0")