import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import zipfile
import random
from ultralytics import YOLO

def main():
    # 1. Setup device and model paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running test on: {str(device).upper()}")
    
    zip_path = 'ucf-dataset.zip'
    model_path = "action_classifier.pth"
    
    if not os.path.exists(zip_path):
        print(f"Error: Dataset zip '{zip_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model weights '{model_path}' not found. Please train first.")
        return

    # 2. Extract a random Burglary test frame directly from ZIP
    print("Selecting a random Burglary test frame from ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as archive:
        all_files = archive.namelist()
        
    # Gather all png frames from the test set of Burglary
    test_burglary_files = [f for f in all_files if f.startswith('Test/Burglary/') and f.endswith('.png')]
    
    if not test_burglary_files:
        print("Error: Could not find Test/Burglary files in ZIP.")
        return
        
    selected_path = random.choice(test_burglary_files)
    print(f"Selected frame: {selected_path}")
    
    # Read the image directly from ZIP in memory
    with zipfile.ZipFile(zip_path, 'r') as archive:
        img_data = archive.read(selected_path)
    
    # Save the original raw frame to disk first
    with open("original_raw_frame.png", "wb") as f:
        f.write(img_data)
    print("Saved raw original image as 'original_raw_frame.png'")

    # Convert bytes to PIL & OpenCV images
    pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # We want to upscale the raw 64x64 frame to a larger size (e.g. 400x400) 
    # so we can easily view the bounding box and text overlays!
    cv_frame = cv2.resize(cv2.imread("original_raw_frame.png"), (480, 480))
    
    # 3. Load YOLOv8
    print("Loading YOLOv8...")
    yolo_model = YOLO("yolov8n.pt")
    
    # 4. Load Action Classifier
    print("Loading Action Classifier...")
    classifier = models.efficientnet_b0()
    num_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(num_features, 2)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    CLASS_NAMES = ["Normal", "Suspicious (Burglary)"]

    # 5. Run Detection on the upscaled image
    # We use YOLOv8 to find the human on the upscaled frame
    results = yolo_model(cv_frame, device=device, classes=[0], verbose=False)
    boxes = results[0].boxes
    
    print(f"YOLO found {len(boxes)} humans in this frame.")
    
    if len(boxes) > 0:
        # Get coordinates of the largest human
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
        
        # Crop and preprocess for the action classifier
        crop = cv_frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        
        input_tensor = preprocess(pil_crop).unsqueeze(0).to(device)
        
        # Run classification
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            conf = probs[pred].item()
            
        label = CLASS_NAMES[pred]
        print(f"AI Classification Result: {label} ({conf*100:.2f}%)")
        
        # Set colors (Red for Burglary/Suspicious, Green for Normal)
        box_color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        
        # Draw bounding box and text label
        cv2.rectangle(cv_frame, (x1, y1), (x2, y2), box_color, 3)
        label_text = f"{label} ({conf*100:.1f}%)"
        
        # Draw text label banner
        text_y = y1 - 10 if y1 > 40 else y1 + 25
        banner_y1 = y1 - 35 if y1 > 40 else y1
        banner_y2 = y1 if y1 > 40 else y1 + 35
        cv2.rectangle(cv_frame, (x1, banner_y1), (x1 + len(label_text)*12, banner_y2), box_color, -1)
        cv2.putText(cv_frame, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw translucent red alert banner at the top of the frame
        overlay = cv_frame.copy()
        cv2.rectangle(overlay, (0, 0), (cv_frame.shape[1], 45), (0, 0, 255) if pred == 1 else (0, 255, 0), -1)
        banner_title = "🚨 SECURE AREA: INTRUDER DETECTED" if pred == 1 else "SECURE AREA: NORMAL ACTIVITY"
        cv2.putText(overlay, banner_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.7, cv_frame, 0.3, 0, cv_frame)
    else:
        # If YOLO did not detect a person in the upscaled 64x64 frame, 
        # classify the whole frame directly!
        print("YOLO box not found, classifying entire frame directly...")
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(probs).item()
            conf = probs[pred].item()
            
        label = CLASS_NAMES[pred]
        print(f"Direct Frame Result: {label} ({conf*100:.2f}%)")
        
        # Draw full border alert
        border_color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        cv2.rectangle(cv_frame, (0, 0), (cv_frame.shape[1], cv_frame.shape[0]), border_color, 10)
        
        # Draw alert text
        cv2.putText(cv_frame, f"{label} ({conf*100:.1f}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
        
    # Save the final annotated result image
    cv2.imwrite("burglary_test_result.jpg", cv_frame)
    print("SUCCESS! Test complete. Saved annotated result as 'burglary_test_result.jpg'!")

if __name__ == "__main__":
    main()
