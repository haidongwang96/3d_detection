# 3d-human-pose-estimation and intrusion detection

Detection：yolov11-pose
2d tracker: deepsort(with mobilenet reid model)

calibration instruction:  
intr：  
record multiple chessboard for each camera  
'tools\preprocessing_tools.py' for intr calibration  
extr：  
record duo camera in the same time  
 mark 4dot, in clockwise for 3d space [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]  
'tools\preprocessing_tools.py' for extr calibration(pnp)  


use 'tools\run.py' to run the program!