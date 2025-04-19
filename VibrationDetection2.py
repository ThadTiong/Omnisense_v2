import cv2
import numpy as np
import time

# ROI settings
roi_x, roi_y, roi_w, roi_h = 160, 25, 345, 425
COLOR_DIFF_THRESHOLD = 0.05  # Sensitivity for chromatic change
AREA_THRESHOLD = 3000        # Total white area needed to consider a pulse
HYSTERESIS_RATIO = 0.5       # Threshold to reset FSM state

# FSM states
STATE_IDLE = 0
STATE_DETECTED = 1

def generate_deformation_mask(baseline, current):
    # Convert to float for normalization
    baseline = baseline.astype(np.float32)
    current = current.astype(np.float32)

    # Normalize RGB vectors (brightness-invariant)
    baseline_norm = baseline / (np.linalg.norm(baseline, axis=2, keepdims=True) + 1e-6)
    current_norm = current / (np.linalg.norm(current, axis=2, keepdims=True) + 1e-6)

    # Compute chromatic difference
    diff = np.linalg.norm(baseline_norm - current_norm, axis=2)

    # Threshold the difference
    _, mask = cv2.threshold(diff, COLOR_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # Clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def detect_pulse_fsm(mask, pulse_state, pulse_count):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)

    print(f"[FSM] Contours: {len(contours)}, Area: {total_area:.1f}, State: {pulse_state}")

    if pulse_state == STATE_IDLE:
        if total_area > AREA_THRESHOLD:
            pulse_count += 1
            pulse_state = STATE_DETECTED
            print("✅ Pulse counted")
    elif pulse_state == STATE_DETECTED:
        if total_area < AREA_THRESHOLD * HYSTERESIS_RATIO:
            pulse_state = STATE_IDLE
            print("↩️  Ready for next pulse")

    return pulse_count, pulse_state

def calculate_bpm(pulse_count, start_time):
    elapsed = time.time() - start_time
    return round((pulse_count / elapsed) * 60, 2) if elapsed > 0 else 0

def calculate_hz(pulse_count, start_time):
    elapsed = time.time() - start_time
    return round(pulse_count / elapsed, 2) if elapsed > 0 else 0

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Capture initial RGB baseline
    ret, initial_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture baseline frame")

    initial_frame = cv2.resize(initial_frame, (640, 480))
    baseline_roi = initial_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Initialize state
    pulse_count = 0
    pulse_state = STATE_IDLE
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        current_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Generate binary mask
        deformation_mask = generate_deformation_mask(baseline_roi, current_roi)

        # Pulse detection using FSM
        pulse_count, pulse_state = detect_pulse_fsm(deformation_mask, pulse_state, pulse_count)

        # Calculate metrics
        bpm = calculate_bpm(pulse_count, start_time)
        hz = calculate_hz(pulse_count, start_time)

        # Display metrics on live feed
        cv2.putText(frame, f"Pulses: {pulse_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"BPM: {bpm}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Hz: {hz}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw ROI on live feed
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Create side-by-side comparison
        mask_bgr = cv2.cvtColor(deformation_mask, cv2.COLOR_GRAY2BGR)
        comparison = np.hstack((baseline_roi, current_roi, mask_bgr))

        # Show outputs
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Baseline | Current | Deformation Mask", comparison)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
