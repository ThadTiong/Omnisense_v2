from roboflow import Roboflow
rf = Roboflow(api_key="B574xCa57kPtj2kjs427")
project = rf.workspace("workspace-5fppr").project("omnisense-electronics-detection")
dataset = project.version(2).download("yolov7")