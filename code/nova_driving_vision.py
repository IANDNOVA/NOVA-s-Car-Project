import pyttsx3
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ✅ Setup speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# ✅ Load YOLO model (detects 80+ objects)
model = YOLO('yolov8n.pt')

# ✅ Open camera
cap = cv2.VideoCapture(0)

# ✅ Set up video recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format for mp4
out = cv2.VideoWriter('nova_vision_output.mp4', fourcc, 20.0, (640, 480))

# ✅ Memory to avoid repeating messages
last_light_status = ""
last_stop_detected = False
last_object_announcement = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed detected.")
        break

    # Resize to match video writer size
    frame = cv2.resize(frame, (640, 480))

    # =========================
    # 1. LANE DETECTION
    # =========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # =========================
    # 2. YOLO OBJECT DETECTION
    # =========================
    results = model(frame, stream=True)

    object_counts = {"person": 0, "car": 0, "truck": 0, "bicycle": 0, "motorbike": 0}
    stop_detected_this_frame = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Count important objects
            if label in object_counts:
                object_counts[label] += 1

            # Draw boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ✅ Stop sign via YOLO
            if label == "stop sign":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "STOP SIGN", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                stop_detected_this_frame = True

    # Speak once when stop sign first appears
    if stop_detected_this_frame and not last_stop_detected:
        engine.say("There’s a stop sign ahead. Prepare to stop.")
        engine.runAndWait()
        last_stop_detected = True
    elif not stop_detected_this_frame:
        last_stop_detected = False

    # =========================
    # 3. TRAFFIC LIGHT DETECTION
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > 300:
        light_status = "RED LIGHT"
        color = (0, 0, 255)
    elif yellow_pixels > 300:
        light_status = "YELLOW LIGHT"
        color = (0, 255, 255)
    elif green_pixels > 300:
        light_status = "GREEN LIGHT"
        color = (0, 255, 0)
    else:
        light_status = ""
        color = (255, 255, 255)

    if light_status and light_status != last_light_status:
        if light_status == "RED LIGHT":
            engine.say("The light is red. Stop.")
        elif light_status == "YELLOW LIGHT":
            engine.say("The light is yellow. Slow down.")
        elif light_status == "GREEN LIGHT":
            engine.say("The light just turned green. You can go.")
        engine.runAndWait()
        last_light_status = light_status

    if light_status:
        cv2.putText(frame, light_status, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # =========================
    # 4. OBJECT COUNTING ANNOUNCEMENT
    # =========================
    current_time = time.time()
    if current_time - last_object_announcement > 5:
        announcements = []
        for obj, count in object_counts.items():
            if count > 0:
                if count == 1:
                    announcements.append(f"one {obj}")
                else:
                    announcements.append(f"{count} {obj}s")

        if announcements:
            sentence = "I see " + ", ".join(announcements) + "."
            engine.say(sentence)
            engine.runAndWait()

        last_object_announcement = current_time

    # =========================
    # 5. WRITE VIDEO OUTPUT
    # =========================
    out.write(frame)

    # =========================
    # DISPLAY WINDOW
    # =========================
    cv2.imshow('NOVA Driving Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()