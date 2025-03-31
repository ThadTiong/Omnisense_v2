import cv2
import glob
import os
import random
import sys
import torch

# Get the current directory of custom_yolov7.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Append the parent directory of yolov7 to the Python path
yolov7_parent_dir = os.path.join(current_dir, 'yolo/yolov7')
sys.path.append(yolov7_parent_dir)

from yolo.yolov7.models.experimental import attempt_load
from yolo.yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from pathlib import Path
from yolo.yolov7.utils.torch_utils import select_device
from yolo.yolov7.utils.plots import plot_one_box

weights_path = 'weights/custom_yolov7_v2.pt'
classNames = ["button", "capacitor", "diode", "resistor", "transistor"]
class_colors = {
    "button": (255, 0, 0),        # Red
    "capacitor": (0, 255, 0),     # Green
    "diode": (0, 0, 255),         # Blue
    "resistor": (255, 255, 0),    # Yellow
    "transistor": (255, 0, 255)   # Magenta
}
conf_threshold = 0.3

source_path = 'yolo/yolov7/Omnisense-Electronics-Detection/test/images/'
image_files = glob.glob(source_path + '*.jpg') + glob.glob(source_path + '*.jpeg') + glob.glob(source_path + '*.png') + glob.glob(source_path + '*.bmp') + glob.glob(source_path + '*.gif')
random_image = random.choice(image_files)

device = select_device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())  # model stride
half = device.type != 'cpu'  # half precision only supported on CUDA

# def process_yolov7(frame):
#     if frame.shape[2] == 3:  # Check if it's a BGR image
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     img = frame.transpose(2, 0, 1)  # Change from HWC to CHW (channels first)
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if device.type != 'cpu' else img.float()  # Convert to half precision FP16 if supported
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0

#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     with torch.no_grad():
#         pred = model(img, augment=False)[0]

#     # Apply NMS
#     pred = non_max_suppression(pred, conf_threshold, 0.4)

#     # Process detections
#     for i, det in enumerate(pred):  # detections per image
#         if det is not None and len(det):
#             # Rescale boxes from img_size to frame size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

#             # Print results
#             for *xyxy, conf, cls in reversed(det):
#                 label = f'{classNames[int(cls)]}'
#                 color = class_colors[classNames[int(cls)]]
#                 plot_one_box(xyxy, frame, label=label, color=color, line_thickness=1)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
#     return frame

def process_yolov7(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3-channel grayscale

    img = gray_frame.transpose(2, 0, 1)  # Change from HWC to CHW (channels first)
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != 'cpu' else img.float()  # Convert to half precision FP16 if supported
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_threshold, 0.5)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # Print results
            for *xyxy, conf, cls in reversed(det):
                label = f'{classNames[int(cls)]}'
                color = class_colors[classNames[int(cls)]]
                plot_one_box(xyxy, frame, label=label, color=color, line_thickness=1)
    return frame

if __name__ == '__main__':
        # Test code
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture('test.mp4')
        #frame = cv2.imread(random_image)
        while True:
            ''' For Input Camera '''
            ret, frame = cap.read()
            if not ret:
                break
            
            ''' For Input Video '''
            #if nn.depthmap_zero_counter >= nn.depthmap_zero_frames - 1:
            #    ret, frame = cap.read()
            #    if not ret:
            #        break
            #
            #if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #    # If yes, reset the video capture to the beginning
            #    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            #''' For Input Image '''
            #if nn.depthmap_zero_counter == nn.depthmap_zero_frames - 1:
            #    frame = cv2.imread('gelsight-samples/sample (1).jpg')

            #height, width = frame.shape[:2]
            #crop_amount = 200
            #frame = frame[:, crop_amount:width-crop_amount]
            frame = cv2.resize(frame, (640, 480))
            
            img = process_yolov7(frame)
            cv2.imshow('YOLOv7', img)

            if cv2.waitKey(int(1000/32)) & 0xFF == ord('q'):
                break
            

        cap.release()
        cv2.destroyAllWindows()