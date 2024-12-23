# Import necessary libraries
import cv2  # For webcam and video processing
from ultralytics import YOLO  # For loading and using the pre-trained model
import torch  # For tensor operations

# Load the pre-trained violence detection model
model_path = "/Users/manojasher/Desktop/best.pt"  # Replace with the correct path to your model
model = YOLO(model_path)  # Automatically loads YOLO models

# Function to capture webcam feed and run real-time detection
def violence_detection_webcam():
    # Initialize webcam (0 = default webcam, adjust index for other cameras)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to exit the webcam feed.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform inference using the loaded model
        results = model.predict(source=frame, save=False, conf=0.5)  # Set confidence threshold (e.g., 0.5)

        # Visualize the results directly on the frame
        annotated_frame = results[0].plot()  # `results[0]` is the first result; `plot()` adds bounding boxes

        # Show the webcam feed with detections
        cv2.imshow("Violence Detection", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    violence_detection_webcam()
