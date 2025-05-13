import sys
import os
sys.path.append("G:/code/3d_detection")  # 添加父目录到搜索路径

import camera

#camera.single_picture_recording(1)
camera.double_picture_recording(0,1,key="video_test")
#camera.double_mp4_recording(0,1, key="test_video")




