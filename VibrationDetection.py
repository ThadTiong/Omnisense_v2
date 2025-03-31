import cv2
import numpy as np
import time

prev_frame = None
last_update_time = time.time()

mm_per_pixel = 0.06

# Define the coordinates of the region of interest (ROI)
roi_x, roi_y, roi_w, roi_h = 160, 25, 345, 425

def preprocess_frame(frame, initial_frame):
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(initial_gray, gray)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, contours

def process_size(frame, pulse_count, pulse_counted):
    lower_threshold = 7
    upper_threshold = 300
    _, contours = preprocess_frame(frame, initial_frame)
    new_box_generated = False
    
    # Draw bounding boxes around contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if lower_threshold <= w <= upper_threshold and lower_threshold <= h <= upper_threshold:
            # Check if the bounding box is within the ROI
            if roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                new_box_generated = True
                
    if new_box_generated and not pulse_counted:
        pulse_count += 1
        pulse_counted = True
    elif not new_box_generated:
        pulse_counted = False
    
    # Draw ROI box
    # cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    return frame, pulse_count, pulse_counted

def calculate_bpm(pulse_count, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 0:
        bpm = (pulse_count / elapsed_time) * 60
        bpm = round(bpm, 2)  # Round to 2 decimal places
    else:
        bpm = 0
    return bpm

def calculate_hz(pulse_count, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 0:
        hz = pulse_count / elapsed_time
        hz = round(hz, 2)  # Round to 2 decimal places
    else:
        hz = 0
    return hz

if __name__ == "__main__":
    # Test code
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Capture the initial frame
    ret, initial_frame = cap.read()
    if not ret:
        raise ValueError("Unable to capture initial frame")

    initial_frame = cv2.resize(initial_frame, (640, 480))
    initial_gray, _ = preprocess_frame(initial_frame, initial_frame)
    
    pulse_count = 0
    pulse_counted = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray, contours = preprocess_frame(frame, initial_frame)
        processed_frame, pulse_count, pulse_counted = process_size(frame, pulse_count, pulse_counted)

        # Draw contours on a black image
        contour_img = np.zeros_like(frame)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Calculate BPM and Hz
        bpm = calculate_bpm(pulse_count, start_time)
        hz = calculate_hz(pulse_count, start_time)

        # Display frames
        cv2.putText(processed_frame, f"Pulses: {pulse_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"BPM: {bpm:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Hz: {hz:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Processed Frame", processed_frame)
        cv2.imshow("Contours", contour_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
