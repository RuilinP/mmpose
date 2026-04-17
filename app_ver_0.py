import csv
import math
import os
import pickle
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk
import sys

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def bundled_path(*parts: str) -> Path:
    """
    For read-only bundled resources:
    - dev: project root
    - PyInstaller: _MEIPASS
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
    else:
        base = Path(__file__).resolve().parent
    return base.joinpath(*parts)


def writable_path(*parts: str) -> Path:
    """
    For runtime output files:
    - dev: project root
    - PyInstaller: folder next to the exe
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    return base.joinpath(*parts)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_POSE = bundled_path("models", "pose_landmarker_full.task")
MODEL_PKL  = bundled_path("models", "rpe_mdl1.pkl")
TEMP_CSV   = writable_path("output", "pose_landmarks_temp.csv")

RECORDING_DURATION_SEC = 60
TARGET_LANDMARK = "LEFT_ANKLE"
CAMERA_ID = 0
DISPLAY_W, DISPLAY_H = 640, 480

LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (27, 29), (29, 31), (27, 31),
    (26, 28), (28, 30), (30, 32), (28, 32),
]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _load_left_ankle_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("landmark_name") != TARGET_LANDMARK:
                continue
            rows.append({
                "frame": int(row["frame"]),
                "time_s": float(row["time_s"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    rows.sort(key=lambda r: r["time_s"])
    return rows


def _compute_velocity(rows: list[dict]) -> list[dict]:
    result = []
    for prev, curr in zip(rows, rows[1:]):
        dt = curr["time_s"] - prev["time_s"]
        if dt <= 0.0:
            continue
        vx = (curr["x"] - prev["x"]) / dt
        vy = (curr["y"] - prev["y"]) / dt
        vz = (curr["z"] - prev["z"]) / dt
        result.append({
            "prev_frame": prev["frame"],
            "frame": curr["frame"],
            "time_s": curr["time_s"],
            "dt": dt,
            "vx": vx, "vy": vy, "vz": vz,
            "magnitude": math.sqrt(vx * vx + vy * vy + vz * vz),
        })
    return result


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _remove_outliers(rows: list[dict], iqr_mult: float = 1.5) -> None:
    if len(rows) < 4:
        return
    mags = sorted(r["magnitude"] for r in rows)
    q1, q3 = _percentile(mags, 0.25), _percentile(mags, 0.75)
    fence = q3 + iqr_mult * (q3 - q1)
    rows[:] = [r for r in rows if r["magnitude"] <= fence]


def _add_minmax_scale(rows: list[dict]) -> None:
    if not rows:
        return
    mags = [r["magnitude"] for r in rows]
    lo, hi = min(mags), max(mags)
    rng = hi - lo
    for r in rows:
        r["magnitude_minmax"] = 0.0 if rng == 0.0 else (r["magnitude"] - lo) / rng


def run_analysis(csv_path: Path) -> float | None:
    """Returns avg magnitude_minmax after outlier removal, or None on failure."""
    ankle = _load_left_ankle_rows(csv_path)
    if not ankle:
        return None
    vel = _compute_velocity(ankle)
    if not vel:
        return None
    _remove_outliers(vel)
    _add_minmax_scale(vel)
    return sum(r["magnitude_minmax"] for r in vel) / len(vel)


def run_prediction(hr: float, vel_mag_scaled: float) -> int:
    with MODEL_PKL.open("rb") as f:
        model = pickle.load(f)
    inputs = {"hr": hr, "velocity_magnitude_scaled": vel_mag_scaled}
    if hasattr(model, "feature_names_in_"):
        x_row = [inputs[n] for n in model.feature_names_in_]
    else:
        x_row = [hr, vel_mag_scaled]
    prediction = int(model.predict([x_row])[0])
    print(f"Model input: {inputs}")
    print(f"Model output: {prediction}")
    return prediction


# ---------------------------------------------------------------------------
# Camera worker thread — captures frames, detects pose, buffers recording rows
# ---------------------------------------------------------------------------

class CameraWorker(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self._lock = threading.Lock()
        self._recording = False
        self._record_start: float = 0.0
        self._landmark_rows: list[dict] = []
        self._stop_event = threading.Event()
        self.on_recording_done: callable | None = None  # set by App

    def start_recording(self) -> None:
        with self._lock:
            self._landmark_rows = []
            self._record_start = time.time()
            self._recording = True

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        base_opts = python.BaseOptions(model_asset_path=str(MODEL_POSE))
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)
        global_start = time.time()
        frame_idx = 0

        try:
            while not self._stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                elapsed_global = time.time() - global_start
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(
                    mp_img, timestamp_ms=int(elapsed_global * 1000)
                )

                # ---- state-sensitive work under lock ----
                recording_done = False
                rows_snapshot: list[dict] = []
                remaining = 0.0

                with self._lock:
                    if self._recording:
                        elapsed_rec = time.time() - self._record_start
                        remaining = max(0.0, RECORDING_DURATION_SEC - elapsed_rec)

                        if result.pose_landmarks:
                            for lm_i, lm in enumerate(result.pose_landmarks[0]):
                                self._landmark_rows.append({
                                    "frame": frame_idx,
                                    "time_s": elapsed_rec,
                                    "landmark": lm_i,
                                    "landmark_name": (
                                        LANDMARK_NAMES[lm_i]
                                        if lm_i < len(LANDMARK_NAMES) else None
                                    ),
                                    "x": lm.x, "y": lm.y, "z": lm.z,
                                    "visibility": getattr(lm, "visibility", float("nan")),
                                    "presence":   getattr(lm, "presence",   float("nan")),
                                })

                        if elapsed_rec >= RECORDING_DURATION_SEC:
                            self._recording = False
                            recording_done = True
                            rows_snapshot = list(self._landmark_rows)

                # ---- draw skeleton ----
                annotated = frame_bgr.copy()
                if result.pose_landmarks:
                    pts = [
                        (int(lm.x * width), int(lm.y * height))
                        for lm in result.pose_landmarks[0]
                    ]
                    for a, b in POSE_CONNECTIONS:
                        if 0 <= a < len(pts) and 0 <= b < len(pts):
                            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
                    for px, py in pts:
                        cv2.circle(annotated, (px, py), 3, (0, 0, 255), -1)

                with self._lock:
                    is_rec = self._recording or recording_done

                if is_rec and not recording_done:
                    cv2.putText(
                        annotated, f"Recording: {remaining:.1f}s",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2,
                    )

                display = cv2.resize(annotated, (DISPLAY_W, DISPLAY_H))
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(display)

                if recording_done:
                    self._save_csv(rows_snapshot)
                    if self.on_recording_done:
                        self.on_recording_done()

                frame_idx += 1
        finally:
            cap.release()
            landmarker.close()

    def _save_csv(self, rows: list[dict]) -> None:
        TEMP_CSV.parent.mkdir(parents=True, exist_ok=True)
        with TEMP_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "time_s", "landmark", "landmark_name",
                "x", "y", "z", "visibility", "presence",
            ])
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main application UI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    _IDLE      = "idle"
    _RECORDING = "recording"
    _ANALYZING = "analyzing"
    _WAIT_HR   = "waiting_hr"
    _RESULT    = "result"

    def __init__(self):
        super().__init__()
        self.title("RPE Fatigue Monitor")
        self.resizable(False, False)
        self._state = self._IDLE
        self._vel_mag_scaled: float = 0.0
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        self._build_ui()

        self._camera = CameraWorker(self._frame_queue)
        self._camera.on_recording_done = lambda: self.after(0, self._do_analysis)
        self._camera.start()
        self._poll_camera()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Camera panel (left)
        self._cam_label = tk.Label(self, bg="black")
        self._cam_label.grid(row=0, column=0, padx=10, pady=10)

        # Control panel (right)
        panel = tk.Frame(self, padx=20, pady=20)
        panel.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            panel, text="RPE Fatigue Monitor",
            font=("Helvetica", 17, "bold"),
        ).pack(pady=(0, 18))

        # Status
        self._status_var = tk.StringVar(value="Press Start to begin.")
        tk.Label(
            panel, textvariable=self._status_var,
            font=("Helvetica", 11), fg="#555555", wraplength=240, justify="center",
        ).pack(pady=(0, 18))

        # Start button
        self._start_btn = tk.Button(
            panel, text="▶  Start Recording",
            font=("Helvetica", 13, "bold"),
            bg="#4CAF50", fg="white", activebackground="#388E3C",
            width=20, pady=6, command=self._on_start,
        )
        self._start_btn.pack(pady=6)

        # HR input (shown only in WAIT_HR state)
        self._hr_frame = tk.Frame(panel)
        tk.Label(
            self._hr_frame, text="Heart Rate (bpm):",
            font=("Helvetica", 11),
        ).pack(anchor="w")
        self._hr_var = tk.StringVar()
        entry_row = tk.Frame(self._hr_frame)
        entry_row.pack(fill="x", pady=(4, 0))
        self._hr_entry = tk.Entry(
            entry_row, textvariable=self._hr_var,
            font=("Helvetica", 14), width=8,
        )
        self._hr_entry.pack(side="left", padx=(0, 8))
        tk.Button(
            entry_row, text="Submit",
            font=("Helvetica", 11, "bold"),
            bg="#1976D2", fg="white", activebackground="#0D47A1",
            padx=8, pady=4, command=self._on_hr_submit,
        ).pack(side="left")
        self._hr_entry.bind("<Return>", lambda _e: self._on_hr_submit())

        # Result display (shown only in RESULT state)
        self._result_frame = tk.Frame(panel, pady=10)
        tk.Label(
            self._result_frame, text="Prediction",
            font=("Helvetica", 12, "bold"),
        ).pack()
        self._result_var = tk.StringVar()
        self._result_label = tk.Label(
            self._result_frame, textvariable=self._result_var,
            font=("Helvetica", 22, "bold"), width=18, pady=10,
            relief="ridge", borderwidth=2,
        )
        self._result_label.pack(pady=8)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Camera feed polling (runs every 30 ms on main thread)
    # ------------------------------------------------------------------
    def _poll_camera(self) -> None:
        try:
            frame_bgr = self._frame_queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(img)
            self._cam_label.configure(image=photo, width=DISPLAY_W, height=DISPLAY_H)
            self._cam_label.image = photo
        except queue.Empty:
            pass
        self.after(30, self._poll_camera)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _set_state(self, state: str) -> None:
        self._state = state

        if state == self._IDLE:
            self._status_var.set("Press Start to begin.")
            self._start_btn.configure(state="normal")
            self._hr_frame.pack_forget()
            self._result_frame.pack_forget()

        elif state == self._RECORDING:
            self._status_var.set(f"Recording for {RECORDING_DURATION_SEC}s…")
            self._start_btn.configure(state="disabled")
            self._hr_frame.pack_forget()
            self._result_frame.pack_forget()

        elif state == self._ANALYZING:
            self._status_var.set("Analysing motion data…")
            self._start_btn.configure(state="disabled")

        elif state == self._WAIT_HR:
            self._status_var.set("Analysis done. Enter your heart rate and press Submit.")
            self._hr_var.set("")
            self._hr_frame.pack(pady=12, fill="x")
            self._hr_entry.focus_set()

        elif state == self._RESULT:
            self._status_var.set("Next recording starts in 5 seconds…")
            self._hr_frame.pack_forget()
            self._result_frame.pack(pady=12)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        self._set_state(self._RECORDING)
        self._camera.start_recording()

    def _do_analysis(self) -> None:
        self._set_state(self._ANALYZING)
        result = run_analysis(TEMP_CSV)
        self._vel_mag_scaled = result if result is not None else 0.0
        self._set_state(self._WAIT_HR)

    def _on_hr_submit(self) -> None:
        try:
            hr = float(self._hr_var.get().strip())
        except ValueError:
            self._status_var.set("Invalid value — please enter a number.")
            return

        try:
            prediction = run_prediction(hr, self._vel_mag_scaled)
        except Exception as exc:
            self._status_var.set(f"Model error: {exc}")
            return

        if prediction == 1:
            self._result_var.set("1 — HIGH FATIGUE")
            self._result_label.configure(fg="white", bg="#c62828")
        else:
            self._result_var.set("0 — NOT HIGH FATIGUE")
            self._result_label.configure(fg="white", bg="#2e7d32")

        self._set_state(self._RESULT)
        # Auto-restart the loop after 5 seconds
        self.after(5000, self._on_start)

    def _on_close(self) -> None:
        self._camera.stop()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    App().mainloop()
