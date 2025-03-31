import cv2
import numpy as np
import time
from os_reconstruction.pcdreconstructor import PointCloudReconstuctor
from os_reconstruction.pcdvisualizer import PointCloudVisualizer
from camera import Camera

''' Define Calibrations '''
target_width = 320*2
target_height = 240*2
threshold = 0.05
object_distance = -62.4
normal_light_vectors = np.array([
    [0, -4.35, 17.35],
    [0, -4.35, -17.35],
    [17.35, -4.35, 0],
])
inverse_light_vectors = np.linalg.pinv(normal_light_vectors)
max_height = 4

left_crop = 20
right_crop = 80
top_crop = 20
bottom_crop = 20
x_crop = left_crop + right_crop
y_crop = top_crop + bottom_crop

''' Initialize Visualizer and Reconstructor '''
nn = PointCloudReconstuctor(target_width - x_crop, target_height - y_crop)
o3d = PointCloudVisualizer(target_width - x_crop, target_height - y_crop)


def generate_heatmap(depthmap):
    """
    Converts the depth map into a 2D heatmap.
    """
    # Normalize depth map to 0-255
    depthmap_normalized = cv2.normalize(depthmap, None, 0, 255, cv2.NORM_MINMAX)
    depthmap_uint8 = np.uint8(depthmap_normalized)

    # Apply a heatmap color mapping
    heatmap = cv2.applyColorMap(depthmap_uint8, cv2.COLORMAP_JET)

    return heatmap


def overlay_text(frame, text, position=(10, 30), color=(0, 255, 0)):
    """
    Overlay text on the frame.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def process_photometricstereo(frame):
    """
    Processes the frame to compute the depth map and generate a heatmap.
    """
    start_time = time.time()

    height, width = frame.shape[:2]
    frame = frame[:, :, ::-1]  # Convert RGB to BGR if needed by OpenCV

    ''' Get depth map using light vectors '''
    depthmap = nn.get_depthmap(frame)

    ''' Generate a 2D heatmap from the depth map '''
    heatmap = generate_heatmap(depthmap)

    ''' Overlay the heatmap onto the original image '''
    overlayed_heatmap = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    end_time = time.time()
    computation_time = end_time - start_time

    #overlay_text(overlayed_heatmap, f"Processing Time: {computation_time:.4f} sec", (10, 30))

    return overlayed_heatmap


IMAGE = 1
VIDEO = 2
CAMERA = 3

if __name__ == '__main__':
    ''' Test Code '''
    input_choice = CAMERA

    if input_choice == IMAGE:
        frame = cv2.imread('gelsight-samples/blank.jpg')
    elif input_choice == VIDEO:
        cap = cv2.VideoCapture('omnisense-samples/sample (12).mp4')
    elif input_choice == CAMERA:
        camera = Camera()
        camera.connect(0, cv2.CAP_DSHOW)

    while True:
        ''' For Input Image '''
        if input_choice == IMAGE:
            if nn.depthmap_zero_counter == nn.depthmap_zero_frames - 1:
                frame = cv2.imread('gelsight-samples/sample (6).jpg')

        ''' For Input Video '''
        if input_choice == VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ''' For Input Camera '''
        if input_choice == CAMERA:
            frame = camera.get_frame()

        frame = cv2.resize(frame, (target_width, target_height))
        frame = frame[top_crop:target_height - bottom_crop, left_crop:target_width - right_crop]

        # Process and display the heatmap overlay
        img = process_photometricstereo(frame)
        cv2.imshow('2D Heatmap Visualization', img)

        if cv2.waitKey(int(1000/32)) & 0xFF == ord('s'):
            o3d.save_pointcloud()

        if cv2.waitKey(int(1000/32)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
