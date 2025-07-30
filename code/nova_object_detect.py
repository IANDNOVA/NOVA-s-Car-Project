from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # "n" = nano (fast and light for MacBook)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed detected.")
        break

    # Run YOLO on the frame
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class name (e.g., "person", "car")
            cls = int(box.cls[0])
            label = model.names[cls]

            # Write each detected object into NOVA_Vision_Log.txt
            with open("NOVA_Vision_Log.txt", "a") as log:
                log.write(f"I see a {label}\n")

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('NOVA Object Detection', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()