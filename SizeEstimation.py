import cv2
import numpy as np

# Fixed conversion factor (adjust based on calibration)
mm_per_pixel = 0.06  

# Store previous bounding box to smooth transitions
prev_rect = None
alpha = 0.7  # Weight for smoothing (higher = more stable, lower = faster response)

# Set individual margins for each side (in pixels)
margin_top = 10
margin_bottom = 10
margin_left = 20
margin_right = 80

def detect_edges_and_measure(frame):
    """
    Detects edges, draws a smoothed rotated bounding box (ignoring per-side border edges), 
    and visually marks the border exclusion zone.
    """
    global prev_rect  # To store previous bounding box for smoothing

    # Get image dimensions
    height, width = frame.shape[:2]

    # Draw border exclusion area (Red rectangle)
    # cv2.rectangle(frame, (margin_left, margin_top), 
    #               (width - margin_right, height - margin_bottom), 
    #               (0, 0, 255), 2)  # Red border

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 20, 100)

    # Dilate edges slightly to make contours more stable
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter contours: Ignore objects near the set borders
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (x > margin_left and x + w < width - margin_right and
                y > margin_top and y + h < height - margin_bottom):
                filtered_contours.append(cnt)

        # Continue only if we have valid contours
        if filtered_contours:
            # Find the largest valid contour by area
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Get the minimum area rectangle (rotated bounding box)
            rect = cv2.minAreaRect(largest_contour)  # (center, (w, h), angle)
            box = cv2.boxPoints(rect)  # Get 4 corner points
            box = np.intp(box)  # Convert to integer format

            # Smooth bounding box transition using weighted average
            if prev_rect is not None:
                # Blend old values with new values to stabilize
                new_center = (
                    int(alpha * prev_rect[0][0] + (1 - alpha) * rect[0][0]),
                    int(alpha * prev_rect[0][1] + (1 - alpha) * rect[0][1])
                )
                new_size = (
                    int(alpha * prev_rect[1][0] + (1 - alpha) * rect[1][0]),
                    int(alpha * prev_rect[1][1] + (1 - alpha) * rect[1][1])
                )
                new_angle = alpha * prev_rect[2] + (1 - alpha) * rect[2]
                rect = (new_center, new_size, new_angle)

            # Update previous bounding box
            prev_rect = rect

            # Convert smoothed rect to box points
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Extract width and height (in pixels) and convert to mm
            w, h = rect[1]  # Width & height of the rotated rectangle
            w_mm = w * mm_per_pixel
            h_mm = h * mm_per_pixel

            # Draw the rotated bounding box
            cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

            # Annotate dimensions
            annotation = f"{w_mm:.2f} x {h_mm:.2f} mm"
            cv2.putText(frame, annotation, (int(rect[0][0]), int(rect[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    return frame, edges

if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check if the camera is accessible
    if not cap.isOpened():
        raise ValueError("Unable to access the camera")

    print("Press 'q' to quit.")

    while True:
        # Capture the current frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Detect edges and measure object
        processed_frame, edge_frame = detect_edges_and_measure(frame)

        # Display frames
        cv2.imshow("Detected Object with Rotated Bounding Box", processed_frame)
        cv2.imshow("Edge Detection", edge_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
