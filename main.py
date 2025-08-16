import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt") 

# Open the video
cap = cv2.VideoCapture("video.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("❌ Error: Video file not found or cannot be opened.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

# Tracking variables
prev_center = None
total_distance = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)[0]

    # Find person class (class 0 in COCO dataset)
    person_boxes = [box for box in results.boxes if int(box.cls.item()) == 0]

    if person_boxes:
        # Choose the largest detected person (by area)
        person_box = max(person_boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])

        # Center point of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw bounding box and center point
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Calculate distance (fixed formula)
        if prev_center is not None:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            distance = math.sqrt(dx**2 + dy**2)  # Correct formula
            total_distance += distance

            # Draw movement line
            cv2.line(frame, prev_center, (cx, cy), (255, 0, 0), 2)

        prev_center = (cx, cy)

    # Show total distance on the frame
    cv2.putText(frame, f"Total Movement: {int(total_distance)} pixels", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write and show video
    out.write(frame)
    cv2.imshow("Tracking", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ✅ Final distance output
print(f"\n✅ Total movement distance: {int(total_distance)} pixels\n")
