import cv2
import numpy as np
import time

# Initialize prevgray outside the function to keep its state between calls
prevgray = None
accumulated_flow = None
last_update_time = time.time()

def preprocess_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and enhance edges
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Calculate the gradient using Sobel operators
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=9)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=9)

    # Combine gradient x and y components
    gradient = cv2.magnitude(grad_x, grad_y)

    # Convert gradient to uint8
    gradient = np.uint8(255 * gradient / np.max(gradient))

    # Apply a binary threshold to make the frame binary
    _, binary_frame = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)

    return binary_frame

def detect_blobs(frame):
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame)

    return keypoints

def categorize_force(fx, fy):
    magnitude_fx = np.mean(np.abs(fx))
    magnitude_fy = np.mean(np.abs(fy))

    positive_fx = np.mean(np.abs(fx[fx < 0]))
    negative_fx = np.mean(np.abs(fx[fx > 0]))
    positive_fy = np.mean(np.abs(fy[fy < 0]))
    negative_fy = np.mean(np.abs(fy[fy > 0]))

    print('fx =', magnitude_fx)
    print('fy =', magnitude_fy)
    print('Positive fx =', positive_fx)
    print('Negative fx =', negative_fx)
    print('Positive fy =', positive_fy)
    print('Negative fy =', negative_fy)

    force_category = ""

    slip_threshold=3
    shear_threshold=5

    if magnitude_fx < slip_threshold and magnitude_fy < slip_threshold:
        force_category = "Normal Force"
    elif positive_fx > shear_threshold and negative_fx > shear_threshold and positive_fy < shear_threshold:
        force_category = "Shear Force"
    elif positive_fy > shear_threshold and negative_fy > shear_threshold and positive_fx < shear_threshold:
        force_category = "Shear Force"
    elif np.abs(magnitude_fx - magnitude_fy) < 4 and magnitude_fx > slip_threshold:
        force_category = "Torque Force"
    elif np.abs(magnitude_fx - magnitude_fy) > slip_threshold and (magnitude_fx > slip_threshold or magnitude_fy > slip_threshold):
        force_category = "Slip Force"
    else:
        force_category = "Normal Force"
    

    return force_category

def process_force(frame):
    global prevgray, accumulated_flow, last_update_time  # Use the global variables

    # Preprocess the cropped frame
    preprocessed_frame = preprocess_frame(frame)

    # Detect blobs in preprocessed frame
    keypoints = detect_blobs(preprocessed_frame)

    if prevgray is not None and keypoints:
        # # Calculate elapsed time since the last update
        # elapsed_time = time.time() - last_update_time

        # if elapsed_time >= 2:
        #     accumulated_flow = 0
        #     last_update_time = time.time()

        # Calculate optical flow using Farneback method on keypoints
        flow = cv2.calcOpticalFlowFarneback(prevgray, preprocessed_frame, None, 0.5, 5, 15, 5, 7, 1.5, 0)

        # Initialize accumulated_flow on the first iteration
        if accumulated_flow is None:
            accumulated_flow = np.zeros_like(flow)

        # Apply median filtering to the optical flow
        filtered_flow = cv2.medianBlur(flow, 5)  # Adjust the kernel size as needed

        # Update the accumulated flow
        accumulated_flow += filtered_flow

        # Extract only significant flow vectors (above a certain threshold)
        significant_indices = np.where(np.sqrt(accumulated_flow[..., 0]**2 + accumulated_flow[..., 1]**2) > 10)
        significant_flow = accumulated_flow[significant_indices]

        # Estimate forces based on significant flow vectors
        fx, fy = significant_flow[..., 0], significant_flow[..., 1]

        # Visualize optical flow on keypoints
        for i, kp in enumerate(keypoints):
            x, y = kp.pt
            dx, dy = accumulated_flow[int(y), int(x)]
            cv2.arrowedLine(frame, (int(x), int(y)), (int(x+dx), int(y+dy)), (0, 255, 0), 2)

        # Estimate forces based on optical flow
        # Extract flow components
        fx, fy = accumulated_flow[..., 0], accumulated_flow[..., 1]

        # Categorize force
        force_category = categorize_force(fx, fy)

        # Display force information on the frame
        cv2.putText(frame, f"Force: {force_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update prevgray for the next iteration
    prevgray = preprocessed_frame.copy()

    return frame


if __name__ == "__main__":
    # Test code
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        processed_frame = process_force(frame)

        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('d'):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
        # elif key == ord('a'):
        #     current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 2))

    cap.release()
    cv2.destroyAllWindows()