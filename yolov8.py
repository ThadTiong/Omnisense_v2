import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv


def process_yolov8(frame):

    global model
    if 'model' not in globals():
        # Lazy load the model if it's not already loaded
        # Object Detection weights should be inputted here
        model = YOLO("weights/yolov8n_640x640.pt")
        model.fuse()

    confidence_threshold = 0.6

    results = model(frame)

    if not results:
        print("No detection results.")
        return frame

    xyxys = []
    confidences = []
    class_ids = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            continue

        class_id = boxes[0].cls
        if class_id in [0.0, 1.0, 2.0, 3.0, 4.0]:
            confidence = result.boxes.conf.cpu().numpy()
            above_threshold = confidence >= confidence_threshold
            filtered_boxes = boxes[above_threshold]
            if len(filtered_boxes) > 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(confidence)
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

    if not xyxys:
        print("No objects found.")
        return frame

    # Setup detections for visualization
    detections = sv.Detections(
        xyxy=np.concatenate(xyxys),
        confidence=np.concatenate(confidences),
        class_id=np.concatenate(class_ids),
    )

    # Print class IDs and confidence scores
    for i, (confidence, class_id) in enumerate(zip(detections.confidence, detections.class_id)):
        print(f"Detection {i + 1}: Class ID: {class_id}, Confidence: {confidence}")

    # Format custom labels
    class_names_dict = model.model.names
    labels = [f"{class_names_dict[class_id]} {confidence:0.2f}"
              for confidence, class_id in zip(detections.confidence, detections.class_id)]

    # Annotate frame
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels,
    )

    return annotated_frame



if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Process every 5th frame
        if count % 1 == 0:
            processed_frame = process_yolov8(frame)
            cv2.imshow("Processed Frame", processed_frame)
        else:
            # Display frame without processing
            cv2.imshow("Processed Frame", frame)

        count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
