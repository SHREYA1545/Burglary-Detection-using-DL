"""
FastAPI Backend for Burglary Detection System.
Provides REST API endpoints for video/image upload analysis,
alert history, and live webcam streaming via WebSocket.

EfficientNet-B0 classifier: TensorFlow/Keras
YOLO human detector       : PyTorch (ultralytics)
"""

import cv2
import torch                          # Still needed for YOLO
import numpy as np                    # For Keras image preprocessing
import tensorflow as tf               # For Keras EfficientNet classifier
import io
import os
import time
import base64
import subprocess
import asyncio
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

import database

# ============================================================
# 1. Initialize AI Models (loaded once at startup)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SERVER] Running on device: {str(device).upper()}")

# Load YOLOv8 (PyTorch — detects humans in frame)
print("[SERVER] Loading YOLOv8 human detector...")
yolo_model = YOLO("yolov8n.pt")

# Load EfficientNet-B0 Action Classifier (Keras — classifies Normal vs Burglary)
print("[SERVER] Loading Action Classifier (EfficientNet-B0 Keras)...")
classifier = tf.keras.models.load_model("action_classifier.keras")

CLASS_NAMES = ["Normal", "Suspicious (Burglary)"]

print("[SERVER] All models loaded successfully!")


# ============================================================
# 2. Keras Image Preprocessing Helper
# ============================================================
def prepare_image_for_keras(bgr_image):
    """
    Convert an OpenCV BGR image to a normalized numpy array
    ready for Keras EfficientNet prediction.
    Input : OpenCV BGR image (any size)
    Output: numpy array of shape (1, 64, 64, 3) with values 0.0-1.0
    """
    rgb_image   = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # BGR → RGB
    resized     = cv2.resize(rgb_image, (64, 64))              # Resize to 64x64 (model input size)
    normalized  = resized.astype(np.float32) / 255.0           # Normalize 0-255 → 0.0-1.0
    batched     = np.expand_dims(normalized, axis=0)           # Add batch dimension → (1,64,64,3)
    return batched


# ============================================================
# 3. Core Detection Function
# ============================================================
def process_frame(frame, consecutive_suspicious_frames):
    """Detect person → Crop → Classify → Track alerts"""
    results = yolo_model(frame, device=device, classes=[0], verbose=False)
    boxes   = results[0].boxes

    person_detected    = len(boxes) > 0
    is_suspicious_frame = False
    current_conf       = 0.0

    if person_detected:
        # Find the largest detected person
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
            # ── Keras classification (replaces PyTorch block) ──
            input_array   = prepare_image_for_keras(crop)
            probabilities = classifier.predict(input_array, verbose=0)[0]
            prediction    = int(np.argmax(probabilities))
            confidence    = float(probabilities[prediction])

            current_conf = confidence

            if prediction == 1 and confidence > 0.75:
                consecutive_suspicious_frames += 1
                is_suspicious_frame = True
                box_color = (0, 0, 255)
            else:
                consecutive_suspicious_frames = max(0, consecutive_suspicious_frames - 1)
                box_color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            label_text = f"{CLASS_NAMES[prediction]} ({confidence*100:.1f}%)"
            text_y     = y1 - 10 if y1 > 40 else y1 + 25
            banner_y1  = y1 - 35 if y1 > 40 else y1
            banner_y2  = y1 if y1 > 40 else y1 + 35
            cv2.rectangle(frame, (x1, banner_y1), (x1 + len(label_text)*12, banner_y2), box_color, -1)
            cv2.putText(frame, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Backup: classify full frame when YOLO misses a person
        # ── Keras classification on full frame ──
        input_array   = prepare_image_for_keras(frame)
        probabilities = classifier.predict(input_array, verbose=0)[0]
        prediction    = int(np.argmax(probabilities))
        confidence    = float(probabilities[prediction])

        current_conf = confidence

        if prediction == 1 and confidence > 0.75:
            consecutive_suspicious_frames += 1
            is_suspicious_frame = True
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
        else:
            consecutive_suspicious_frames = max(0, consecutive_suspicious_frames - 1)

    # Draw alert banner at top of frame
    alert_triggered = consecutive_suspicious_frames > 20
    overlay = frame.copy()
    if alert_triggered:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(overlay, f"BURGLARY ALERT! Suspicious frames: {consecutive_suspicious_frames}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)
        cv2.putText(overlay, f"System Active | Frames: {consecutive_suspicious_frames}/20",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    return frame, consecutive_suspicious_frames, is_suspicious_frame, current_conf


def play_alarm():
    """Play security alarm sound asynchronously on Windows."""
    alarm_file = "security-alarm.mp3"
    if os.path.exists(alarm_file):
        abs_path = os.path.abspath(alarm_file)
        cmd = f"powershell -c \"Add-Type -AssemblyName PresentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open('{abs_path}'); $player.Play(); Start-Sleep -s 15\""
        subprocess.Popen(cmd, shell=True)


# ============================================================
# 4. FastAPI Application
# ============================================================
app = FastAPI(title="Burglary Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")  # Serve annotated videos

# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/status")
async def get_status():
    """Health check endpoint."""
    return {"status": "online", "device": str(device).upper()}


@app.post("/api/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Upload a video (.mp4) or image (.jpg/.png) file.
    Runs the full detection pipeline and returns the analysis report.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png']:
        return JSONResponse(status_code=400, content={"error": f"Unsupported file format: {file_ext}"})

    save_path = os.path.join("uploads", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    is_image = file_ext in ['.jpg', '.jpeg', '.png']

    if is_image:
        frame = cv2.imread(save_path)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image file."})

        _, _, is_susp, conf = process_frame(frame, 25)
        verdict = "SUSPICIOUS" if is_susp else "NORMAL"

        database.insert_alert(
            source=file.filename, total_frames=1,
            suspicious_frames=1 if is_susp else 0,
            suspicious_ratio=1.0 if is_susp else 0.0,
            peak_confidence=conf, verdict=verdict,
            alarm_triggered=is_susp
        )

        if is_susp:
            play_alarm()

        os.remove(save_path)

        return {
            "source": file.filename, "type": "image",
            "total_frames": 1,
            "suspicious_frames": 1 if is_susp else 0,
            "suspicious_ratio": 100.0 if is_susp else 0.0,
            "peak_confidence": round(conf * 100, 2),
            "verdict": verdict, "alarm_triggered": is_susp
        }
    else:
        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Could not open video file."})

        # Get video properties for output writer
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create annotated output video file
        annotated_filename = f"annotated_{os.path.splitext(file.filename)[0]}.mp4"
        annotated_path     = os.path.join("uploads", annotated_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        total_frames            = 0
        suspicious_frames_count = 0
        peak_confidence         = 0.0
        consecutive             = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            annotated_frame, consecutive, is_susp, conf = process_frame(frame, consecutive)
            out.write(annotated_frame)  # Write annotated frame to output video

            if is_susp:
                suspicious_frames_count += 1
                if conf > peak_confidence:
                    peak_confidence = conf

        cap.release()
        out.release()   # Finalize annotated video file
        os.remove(save_path)

        suspicious_ratio    = suspicious_frames_count / total_frames if total_frames > 0 else 0
        is_video_suspicious = suspicious_ratio > 0.05 and peak_confidence > 0.75
        verdict             = "SUSPICIOUS" if is_video_suspicious else "NORMAL"

        database.insert_alert(
            source=file.filename, total_frames=total_frames,
            suspicious_frames=suspicious_frames_count,
            suspicious_ratio=suspicious_ratio,
            peak_confidence=peak_confidence, verdict=verdict,
            alarm_triggered=is_video_suspicious
        )

        if is_video_suspicious:
            play_alarm()

        return {
            "source": file.filename, "type": "video",
            "total_frames": total_frames,
            "suspicious_frames": suspicious_frames_count,
            "suspicious_ratio": round(suspicious_ratio * 100, 2),
            "peak_confidence": round(peak_confidence * 100, 2),
            "verdict": verdict, "alarm_triggered": is_video_suspicious,
            "annotated_video_url": f"/uploads/{annotated_filename}"
        }


@app.get("/api/alerts")
async def get_alerts():
    """Get all alert history from the database."""
    alerts = database.get_all_alerts()
    return {"alerts": alerts, "total": len(alerts)}


@app.delete("/api/alerts")
async def clear_alerts():
    """Clear all alert history."""
    conn = database.get_connection()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return {"message": "All alerts cleared."}


# ============================================================
# WebSocket: Live Webcam Stream
# ============================================================
@app.websocket("/ws/webcam")
async def webcam_stream(websocket: WebSocket):
    """Stream live webcam feed with detection annotations via WebSocket."""
    await websocket.accept()
    print("[SERVER] WebSocket client connected for webcam stream.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_json({"error": "Could not open webcam."})
        await websocket.close()
        return

    consecutive   = 0
    alarm_playing = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, consecutive, is_susp, conf = process_frame(frame, consecutive)

            if consecutive > 20 and not alarm_playing:
                play_alarm()
                alarm_playing = True
            elif consecutive == 0:
                alarm_playing = False

            _, buffer   = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64   = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "frame": frame_b64,
                "suspicious_frames": consecutive,
                "alert": consecutive > 20
            })

            await asyncio.sleep(0.066)

    except WebSocketDisconnect:
        print("[SERVER] WebSocket client disconnected.")
    finally:
        cap.release()
        print("[SERVER] Webcam released.")


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("[SERVER] Starting Burglary Detection API at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
