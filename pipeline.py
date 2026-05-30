"""
Standalone Burglary Detection Pipeline (CLI usage).
Run from terminal: python pipeline.py [video/image path]

EfficientNet-B0 classifier: TensorFlow/Keras
YOLO human detector       : PyTorch (ultralytics)
"""

import cv2
import torch                  # For YOLO
import numpy as np            # For Keras preprocessing
import tensorflow as tf       # For Keras EfficientNet
import sys
import os
import time
import subprocess
from ultralytics import YOLO


# ============================================================
# 1. Load Models
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running pipeline on: {str(device).upper()}")

print("Loading YOLOv8 human detector...")
yolo_model = YOLO("yolov8n.pt")

print("Loading Action Classifier (EfficientNet-B0 Keras)...")
model_path = "action_classifier.keras"
if not os.path.exists(model_path):
    print(f"Error: '{model_path}' not found! Train the model on Colab first.")
    sys.exit(1)

classifier = tf.keras.models.load_model(model_path)
CLASS_NAMES = ["Normal", "Suspicious (Burglary)"]
print("All models loaded!\n")


# ============================================================
# 2. Keras Image Preprocessing Helper
# ============================================================
def prepare_image_for_keras(bgr_image):
    """
    Convert OpenCV BGR image → normalized numpy array for Keras prediction.
    Input : OpenCV BGR image (any size)
    Output: numpy array (1, 224, 224, 3) with values 0.0-1.0
    """
    rgb_image  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized    = cv2.resize(rgb_image, (64, 64))
    normalized = resized.astype(np.float32) / 255.0
    batched    = np.expand_dims(normalized, axis=0)
    return batched


# ============================================================
# 3. Alarm
# ============================================================
def play_alarm():
    alarm_file = "security-alarm.mp3"
    if os.path.exists(alarm_file):
        abs_path = os.path.abspath(alarm_file)
        print("🔊 Playing security siren sound...")
        cmd = f"powershell -c \"Add-Type -AssemblyName PresentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open('{abs_path}'); $player.Play(); Start-Sleep -s 15\""
        subprocess.Popen(cmd, shell=True)
    else:
        print("⚠️ Warning: security-alarm.mp3 not found.")


# ============================================================
# 4. Core Detection Function
# ============================================================
def process_frame(frame, consecutive_suspicious_frames):
    """
    Core pipeline: Detect person → Crop → Classify action → Track sliding window alert
    """
    results = yolo_model(frame, device=device, classes=[0], verbose=False)
    boxes   = results[0].boxes

    person_detected     = len(boxes) > 0
    current_label       = "No Person"
    current_conf        = 0.0
    box_color           = (0, 255, 0)
    is_suspicious_frame = False

    if person_detected:
        # Find largest detected person
        largest_box = None
        max_area    = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area    = area
                largest_box = box

        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        h, w, _         = frame.shape
        x1, y1          = max(0, x1), max(0, y1)
        x2, y2          = min(w, x2), min(h, y2)
        crop            = frame[y1:y2, x1:x2]

        if crop.size > 0:
            # ── Keras classification on cropped person ──
            input_array   = prepare_image_for_keras(crop)
            probabilities = classifier.predict(input_array, verbose=0)[0]
            prediction    = int(np.argmax(probabilities))
            confidence    = float(probabilities[prediction])

            current_label = CLASS_NAMES[prediction]
            current_conf  = confidence

            if prediction == 1 and confidence > 0.75:
                consecutive_suspicious_frames += 1
                box_color           = (0, 0, 255)
                is_suspicious_frame = True
            else:
                consecutive_suspicious_frames = max(0, consecutive_suspicious_frames - 1)
                box_color = (0, 255, 0)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            label_text = f"{current_label} ({confidence*100:.1f}%)"
            text_y     = y1 - 10 if y1 > 40 else y1 + 25
            banner_y1  = y1 - 35 if y1 > 40 else y1
            banner_y2  = y1 if y1 > 40 else y1 + 35
            cv2.rectangle(frame, (x1, banner_y1), (x1 + len(label_text)*12, banner_y2), box_color, -1)
            cv2.putText(frame, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Backup: classify full frame if YOLO missed person
        # ── Keras classification on full frame ──
        input_array   = prepare_image_for_keras(frame)
        probabilities = classifier.predict(input_array, verbose=0)[0]
        prediction    = int(np.argmax(probabilities))
        confidence    = float(probabilities[prediction])

        current_label = CLASS_NAMES[prediction]
        current_conf  = confidence

        if prediction == 1 and confidence > 0.75:
            consecutive_suspicious_frames += 1
            is_suspicious_frame = True
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
        else:
            consecutive_suspicious_frames = max(0, consecutive_suspicious_frames - 1)

    # Draw alert banner
    alert_triggered = consecutive_suspicious_frames > 20
    overlay = frame.copy()
    if alert_triggered:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(overlay, f"🚨 BURGLARY ALERT! Suspicious frames: {consecutive_suspicious_frames}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)
        cv2.putText(overlay, f"System Active | Alert Status: Normal (Frames: {consecutive_suspicious_frames}/20)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    return frame, consecutive_suspicious_frames, is_suspicious_frame, current_conf


# ============================================================
# 5. Main — Handle Image / Video / Webcam
# ============================================================
def main():
    input_source = 0
    is_image     = False

    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        if os.path.exists(source_path):
            file_ext = os.path.splitext(source_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png']:
                input_source = source_path
                is_image     = True
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                input_source = source_path
            else:
                print(f"Unsupported file format: {file_ext}")
                sys.exit(1)
        else:
            print(f"Error: Path '{source_path}' does not exist!")
            sys.exit(1)

    consecutive_suspicious_frames = 0

    if is_image:
        print(f"Processing image: {input_source}")
        frame = cv2.imread(input_source)
        if frame is None:
            print("Error: Could not read image.")
            sys.exit(1)

        processed, _, is_susp, conf = process_frame(frame, 25)

        print("\n" + "="*45)
        print("         IMAGE ANALYSIS CONCLUSION          ")
        print("="*45)
        print(f"Image Source: {os.path.basename(input_source)}")
        verdict = "🚨 SUSPICIOUS (Burglary Pose Detected)" if is_susp else "✅ NORMAL (No Burglary Detected)"
        print(f"Final Verdict: {verdict}")
        print(f"Confidence   : {conf*100:.2f}%")
        print("="*45 + "\n")

        cv2.imshow("Burglary Detection - Image Mode", processed)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        is_webcam   = (input_source == 0)
        source_name = "Webcam Feed" if is_webcam else f"Video File ({os.path.basename(input_source)})"
        print(f"Starting capture from: {source_name}")

        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            sys.exit(1)

        print("\nPipeline running! Press 'q' to quit.\n")

        total_frames            = 0
        suspicious_frames_count = 0
        peak_suspicious_confidence = 0.0
        alarm_playing           = False
        start_time_proc         = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            annotated_frame, consecutive_suspicious_frames, is_suspicious, confidence = process_frame(frame, consecutive_suspicious_frames)

            if consecutive_suspicious_frames > 20 and not alarm_playing:
                play_alarm()
                alarm_playing = True
            elif consecutive_suspicious_frames == 0:
                alarm_playing = False

            if is_suspicious:
                suspicious_frames_count += 1
                if confidence > peak_suspicious_confidence:
                    peak_suspicious_confidence = confidence

            cv2.imshow("Burglary Detection System - Live Feed", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video stream completed.")

        if not is_webcam and total_frames > 0:
            processing_duration = time.time() - start_time_proc
            suspicious_ratio    = suspicious_frames_count / total_frames
            is_video_suspicious = suspicious_ratio > 0.05 and peak_suspicious_confidence > 0.75

            print("\n" + "="*50)
            print("       COMPREHENSIVE VIDEO ANALYSIS REPORT        ")
            print("="*50)
            print(f"Video Source      : {os.path.basename(input_source)}")
            print(f"Total Frames      : {total_frames}")
            print(f"Suspicious Frames : {suspicious_frames_count} ({suspicious_ratio*100:.2f}%)")
            print(f"Peak Alert Conf.  : {peak_suspicious_confidence*100:.2f}%")
            print(f"Scan Duration     : {processing_duration:.2f} seconds")
            print("-"*50)

            if is_video_suspicious:
                print("FINAL CONCLUSION : 🚨 SUSPICIOUS (Burglary Activity Detected!)")
                play_alarm()
            else:
                print("FINAL CONCLUSION : ✅ NORMAL (No Suspicious Activity Detected)")
            print("="*50 + "\n")


if __name__ == "__main__":
    main()
