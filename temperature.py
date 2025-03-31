import cv2
import numpy as np

lower_factor = 30 
upper_factor = 30

colors_ranges = [
    {"lower": np.array([150,170,170]) - (lower_factor/6), "upper": np.array([150,170,170]) + (upper_factor/6)},  # Yellow  
    {"lower": np.array([160,100,80]) - (lower_factor), "upper": np.array([160,100,80]) + (upper_factor)},  # Light Blue 
    {"lower": np.array([120,95,135]) - (lower_factor), "upper": np.array([120,95,135]) + (upper_factor)},  # Magenta 
    {"lower": np.array([155,105,70]) - (lower_factor), "upper": np.array([155,105,70]) + (upper_factor)},  # Royal Blue 
    {"lower": np.array([140,140,190]) - (lower_factor), "upper": np.array([140,140,190]) + (upper_factor)},  # Orange 
]

rois = [
    {"coords": [(96, 32), (160, 96)], "label": "O"},  # Define the coordinates of ROI 1
    {"coords": [(96, 128), (160, 192)], "label": "RB"},  # Define the coordinates of ROI 2
    {"coords": [(96, 224), (160, 288)], "label": "M"},  # Define the coordinates of ROI 3
    {"coords": [(96, 320), (160, 384)], "label": "LB"},  # Define the coordinates of ROI 4
    {"coords": [(96, 416), (160, 480)], "label": "Y"}   # Define the coordinates of ROI 5
]

def is_orange_present_in_roi(frame, roi_coords):
    orange_range = colors_ranges[4]  # Orange color range
    return is_color_present_in_roi(frame, orange_range, roi_coords)

def is_royal_blue_present_in_roi(frame, roi_coords):
    royal_blue_range = colors_ranges[3]  # Royal Blue color range
    return is_color_present_in_roi(frame, royal_blue_range, roi_coords)

def is_magenta_present_in_roi(frame, roi_coords):
    magenta_range = colors_ranges[2]  # Magenta color range
    return is_color_present_in_roi(frame, magenta_range, roi_coords)

def is_light_blue_present_in_roi(frame, roi_coords):
    light_blue_range = colors_ranges[1]  # Light Blue color range
    return is_color_present_in_roi(frame, light_blue_range, roi_coords)

def is_yellow_present_in_roi(frame, roi_coords):
    yellow_range = colors_ranges[0]  # Yellow color range
    return is_color_present_in_roi(frame, yellow_range, roi_coords)

def is_color_present_in_roi(frame, color_range, roi_coords):
    x1, y1 = roi_coords[0]  # Top-left coordinate of ROI
    x2, y2 = roi_coords[1]  # Bottom-right coordinate of ROI

    # Check if ROI coordinates are within the bounds of the frame
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return False  # Return False if ROI is out of bounds

    roi_frame = frame[y1:y2, x1:x2]  # Extract ROI from the frame
    
    # Check if the extracted ROI is empty
    if roi_frame.size == 0:
        return False  # Return False if ROI is empty

    mask = cv2.inRange(roi_frame, color_range["lower"], color_range["upper"])  # Create mask for color range in ROI
    return np.any(mask)

def create_color_image(color_range):
    color_image_lower = np.zeros((50, 100, 3), dtype=np.uint8)
    color_image_upper = np.zeros((50, 100, 3), dtype=np.uint8)

    color_image_lower[:, :] = color_range["lower"]
    color_image_upper[:, :] = color_range["upper"]

    return np.vstack([color_image_upper, color_image_lower])

def process_temperature(frame):

    # # Draw rectangles for each ROI
    # for roi in rois:
    #     top_left, bottom_right = roi["coords"]
    #     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw a green rectangle
    
    # Overlay the combined mask on the original frame
    #frame_with_mask = cv2.addWeighted(frame, 0, combined_mask, 1, 0)
    
    # Check for the presence of each color in the lower portion
    orange_presence = is_orange_present_in_roi(frame, rois[0]["coords"])
    royal_blue_presence = is_royal_blue_present_in_roi(frame, rois[1]["coords"])
    magenta_presence = is_magenta_present_in_roi(frame, rois[2]["coords"])
    light_blue_presence = is_light_blue_present_in_roi(frame, rois[3]["coords"])
    yellow_presence = is_yellow_present_in_roi(frame, rois[4]["coords"])

    # Check for specific color combinations
    if orange_presence and royal_blue_presence and magenta_presence and light_blue_presence and yellow_presence:
        cv2.putText(frame, "T < 15", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif orange_presence and royal_blue_presence and magenta_presence and light_blue_presence:
        cv2.putText(frame, "15 < T < 22", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif orange_presence and royal_blue_presence and magenta_presence:
        cv2.putText(frame, "22 < T < 31", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif orange_presence and royal_blue_presence:
        cv2.putText(frame, "31 < T < 38", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif orange_presence:
        cv2.putText(frame, "38 < T < 60", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "60 < T", (320, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame #, frame_with_mask

if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    #cap = cv2.VideoCapture('test.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            processed_frame = process_temperature(frame)

            cv2.imshow("Processed Frame", processed_frame)
            #cv2.imshow("Mask Frame", frame_with_mask)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("Error loading the frame.")
            break

    cap.release()
    cv2.destroyAllWindows()