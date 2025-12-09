import cv2
import numpy as np
import onnxruntime as ort
import time

# ===== CONFIG =====
MODEL_PATH = r"C:\Users\Dararith\Desktop\Fall_2025\Robotic_submission\onnxconversion\model.onnx"

LABELS = {
    1: "Expired_Cosmetic",
    2: "Syringe",
    3: "Tablet",
    4: "Used_Battery"
}

SCORE_THRESHOLD = 0.50

# ===== LOAD MODEL =====
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX input:", input_name)

# ===== START CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✔ Webcam connected. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Model expects RGB uint8
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

    # Run inference
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed = time.time() - start

    # Prevent division by zero
    if elapsed <= 0:
         fps = 999.0
    else:
        fps = 1.0 / elapsed

    # Extract outputs
    boxes = outputs[1][0]            # detection_boxes
    classes = outputs[2][0].astype(int)  # detection_classes
    scores = outputs[4][0]           # detection_scores
    num = int(outputs[5][0])         # num_detections

    # Loop detections
    for i in range(num):
        score = scores[i]
        if score < SCORE_THRESHOLD:
            continue

        y1, x1, y2, x2 = boxes[i]

        # Convert normalized coords to pixel coords
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        class_id = classes[i]
        label = LABELS.get(class_id, "Unknown")

        # Draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("ONNX Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
