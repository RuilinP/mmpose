import os
import cv2
import numpy as np
import pandas as pd
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def run_pose_landmarker_on_live_stream(
    model_path: str,
    out_csv_path: str,
    camera_id: int = 0,
):
    landmark_names = [
        "NOSE",
        "LEFT_EYE_INNER",
        "LEFT_EYE",
        "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER",
        "RIGHT_EYE",
        "RIGHT_EYE_OUTER",
        "LEFT_EAR",
        "RIGHT_EAR",
        "MOUTH_LEFT",
        "MOUTH_RIGHT",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_PINKY",
        "RIGHT_PINKY",
        "LEFT_INDEX",
        "RIGHT_INDEX",
        "LEFT_THUMB",
        "RIGHT_THUMB",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]
    
    RIGHT_HEEL_INDEX = 29  # Index for RIGHT_HEEL in landmark_names
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Storage for landmarks data
    rows = []

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera_id}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # PoseLandmarker options with VIDEO mode (effectively LIVE_STREAM with camera)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    # Define pose connections
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (27, 29), (29, 31), (27, 31),
        (26, 28), (28, 30), (30, 32), (28, 32),
    ]
    
    # Capture frames and send to landmarker
    frame_idx = 0
    start_time = time.time()
    print("Starting live stream pose detection. Press 'q' to quit.")
    
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # timestamp must be in milliseconds, monotonically increasing
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # Detect pose in frame
            result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
            
            # Process results
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]  # first person
                
                # Output RIGHT_HEEL coordinates
                if len(landmarks) > RIGHT_HEEL_INDEX:
                    right_heel = landmarks[RIGHT_HEEL_INDEX]
                    print(f"RIGHT_HEEL - X: {right_heel.x:.4f}, Y: {right_heel.y:.4f}, Z: {right_heel.z:.4f}")
                
                # Store all landmarks
                for lm_i, lm in enumerate(landmarks):
                    rows.append({
                        "timestamp_ms": timestamp_ms,
                        "landmark": lm_i,
                        "landmark_name": landmark_names[lm_i] if lm_i < len(landmark_names) else None,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": getattr(lm, "visibility", np.nan),
                        "presence": getattr(lm, "presence", np.nan),
                    })
            
            # Create annotated frame for display
            annotated = frame_bgr.copy()
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                pts = []
                for lm in landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    pts.append((px, py))
                
                # Draw connections
                for a, b in pose_connections:
                    if 0 <= a < len(pts) and 0 <= b < len(pts):
                        cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
                
                # Draw all landmarks
                for (px, py) in pts:
                    cv2.circle(annotated, (px, py), 3, (0, 0, 255), -1)
                
                # Highlight RIGHT_HEEL landmark in blue
                if RIGHT_HEEL_INDEX < len(pts):
                    px, py = pts[RIGHT_HEEL_INDEX]
                    cv2.circle(annotated, (px, py), 8, (255, 0, 0), -1)  # Blue circle
                    cv2.putText(annotated, "RIGHT_HEEL", (px + 10, py - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display the annotated frame
            cv2.imshow("Pose Landmark Detection (LIVE_STREAM)", annotated)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    # Save landmarks to CSV
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        print(f"Saved {len(rows)} landmark entries to: {out_csv_path}")
    else:
        print("No landmarks detected during live stream session.")


if __name__ == "__main__":
    MODEL = "models/pose_landmarker_full.task"
    OUT_CSV = "output/pose_landmarks_live.csv"

    run_pose_landmarker_on_live_stream(
        model_path=MODEL,
        out_csv_path=OUT_CSV,
        camera_id=0,  # Default webcam (change to 1, 2, etc. for other cameras)
    )