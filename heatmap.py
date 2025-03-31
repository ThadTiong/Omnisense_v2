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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate the gradient using Sobel operators
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradient x and y components
    gradient = cv2.magnitude(grad_x, grad_y)

    # Convert gradient to uint8
    gradient = np.uint8(255 * gradient / np.max(gradient))

    # Apply dilation to enhance features
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.dilate(gradient, kernel, iterations=1)

    return gradient

def generate_heatmap(flow):
    # Calculate the magnitude of the flow vectors
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    heatmap = np.uint8(magnitude)

    # Apply a colormap for visualization
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    return heatmap

def process_heatmap(frame):
    global prevgray, accumulated_flow, last_update_time  # Use the global variables

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    if prevgray is not None:
        # Calculate elapsed time since the last update
        elapsed_time = time.time() - last_update_time

        if elapsed_time >= 2:
            accumulated_flow *= 0.1
            last_update_time = time.time()

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prevgray, preprocessed_frame, None, 0.5, 5, 15, 5, 7, 1.5, 0)

        # Initialize accumulated_flow on the first iteration
        if accumulated_flow is None:
            accumulated_flow = flow
        else:
            # Accumulate optical flow information
            accumulated_flow += flow

        # Apply the optical flow visualization
        # h, w = frame.shape[:2]
        # y, x = np.mgrid[5:h:10, 5:w:10].reshape(2, -1)
        # fx, fy = accumulated_flow[y, x].T
        # lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2).astype(int)

        # for (x1, y1), (x2, y2) in lines:
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #     cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

        # Generate and overlay the heatmap
        heatmap = generate_heatmap(accumulated_flow)
        frame = cv2.addWeighted(frame, 0.5, heatmap, 1, 0)

    # Update prevgray for the next iteration
    prevgray = preprocessed_frame.copy()

    return frame

if __name__ == "__main__":
    # Test code
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_heatmap(frame)

        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
