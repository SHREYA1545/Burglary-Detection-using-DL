# pyrefly: ignore [missing-import]
import cv2
from ultralytics import YOLO
import torch

def main():
    # 1. Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 2. Load the pre-trained YOLOv8 nano model
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # 3. Open the default laptop webcam (ID 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure your camera is connected and not in use by another app.")
        return
        
    print("\nWebcam started successfully!")
    print("Press 'q' key on your keyboard while focusing the video window to quit.\n")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # 4. Run YOLOv8 on the frame (forcing it to use our selected device: cuda/cpu)
        # classes=[0] tells YOLO to ONLY detect person (COCO class 0 is 'person')
        results = model(frame, device=device, classes=[0], verbose=False)
        
        # 5. Extract results and draw bounding boxes on the frame
        annotated_frame = results[0].plot()
        
        # 6. Display the frame in a window
        cv2.imshow("Burglary Detection System - YOLO Webcam Test", annotated_frame)
        
        # Wait for 1ms, and check if the user pressed the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up and release the camera
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed safely.")

if __name__ == "__main__":
    main()
