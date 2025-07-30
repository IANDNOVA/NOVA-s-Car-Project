import cv2

# Open the MacBook camera (0 = built-in)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed detected.")
        break

    cv2.imshow('NOVA Vision Test', frame)

    # Press Q to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
