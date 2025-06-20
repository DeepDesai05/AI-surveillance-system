from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r"C:\nuv\AI_SURVILANCE_05\best (6).pt")

# Open the webcam (0 is default camera; change to 1, 2, etc. if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Run loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(source=frame, conf=0.3, imgsz=640, show=False, stream=True)

    # Draw boxes on the frame
    for r in results:
        annotated_frame = r.plot()

    # Display output
    cv2.imshow("AI Surveillance - Live", annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
