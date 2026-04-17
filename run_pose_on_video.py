import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def run_pose_landmarker_on_video(
    video_path: str,
    model_path: str,
    out_csv_path: str,
    out_overlay_video_path: str | None = None,
    max_frames: int | None = None,
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
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    total = frame_count if frame_count > 0 else None
    if max_frames is not None:
        total = min(total, max_frames) if total is not None else max_frames

    # PoseLandmarker options
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Optional overlay video writer
    writer = None
    if out_overlay_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_overlay_video_path, fourcc, fps, (width, height))
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

    rows = []
    idx = 0

    pbar = tqdm(total=total if total is not None else 0, unit="frame")
    while True:
        if max_frames is not None and idx >= max_frames:
            break

        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # timestamp must be in milliseconds, monotonically increasing
        timestamp_ms = int((idx / fps) * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        # result.pose_landmarks is a list (one per detected person)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]  # first person
            # landmarks: list of NormalizedLandmark (x,y,z,visibility,presence)
            for lm_i, lm in enumerate(landmarks):
                rows.append({
                    "frame": idx,
                    "time_s": idx / fps,
                    "landmark": lm_i,
                    "landmark_name": landmark_names[lm_i] if lm_i < len(landmark_names) else None,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": getattr(lm, "visibility", np.nan),
                    "presence": getattr(lm, "presence", np.nan),
                })

            # Overlay (optional)
            if writer is not None:
                annotated = frame_bgr.copy()

                # Create a fake "pose_landmarks" proto-like object for drawing_utils
                # We'll draw using landmark list + POSE_CONNECTIONS.
                # drawing_utils expects a NormalizedLandmarkList-like object; easiest is manual drawing.
                # We'll draw points + lines ourselves.
                pts = []
                for lm in landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    pts.append((px, py))

                # draw connections
                for a, b in pose_connections:
                    if 0 <= a < len(pts) and 0 <= b < len(pts):
                        cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)

                # draw points
                for (px, py) in pts:
                    cv2.circle(annotated, (px, py), 3, (0, 0, 255), -1)

                writer.write(annotated)

        idx += 1
        if total is not None:
            pbar.update(1)
        else:
            # unknown total; still update visually
            pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved landmarks: {out_csv_path}")
    if out_overlay_video_path:
        print(f"Saved overlay video: {out_overlay_video_path}")


if __name__ == "__main__":
    VIDEO = "input/recording_10_58_24_gmt+8.mp4"
    MODEL = "models/pose_landmarker_full.task"
    OUT_CSV = "output/pose_landmarks_10_58_24.csv"
    OUT_OVERLAY = "output/pose_overlay.mp4"  # set to None if you don't need it

    run_pose_landmarker_on_video(
        video_path=VIDEO,
        model_path=MODEL,
        out_csv_path=OUT_CSV,
        out_overlay_video_path=OUT_OVERLAY,
        max_frames=None,  # e.g. 300 for quick test
    )