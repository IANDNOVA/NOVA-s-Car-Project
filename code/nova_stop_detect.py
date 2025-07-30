import cv2

# Load stop sign cascade (pre-trained Haar model from OpenCV)
stop_sign_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "stop_sign.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed detected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect stop signs
    stops = stop_sign_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in stops:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "STOP SIGN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('NOVA Stop Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
