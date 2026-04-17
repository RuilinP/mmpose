import csv
import math
import pickle
from pathlib import Path


INPUT_CSV = Path(__file__).with_name("pose_landmarks.csv")
OUTPUT_CSV = Path(__file__).with_name("left_ankle_velocity.csv")
MODEL_PKL = Path(__file__).resolve().parent.parent / "models" / "rpe_mdl1.pkl"
TARGET_LANDMARK = "LEFT_ANKLE"


def load_left_ankle_rows(csv_path: Path) -> list[dict]:
	"""Read pose rows and keep only LEFT_ANKLE entries, ordered by time."""
	rows: list[dict] = []

	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if row.get("landmark_name") != TARGET_LANDMARK:
				continue

			rows.append(
				{
					"frame": int(row["frame"]),
					"time_s": float(row["time_s"]),
					"x": float(row["x"]),
					"y": float(row["y"]),
					"z": float(row["z"]),
				}
			)

	rows.sort(key=lambda r: r["time_s"])
	return rows


def compute_velocity(rows: list[dict]) -> list[dict]:
	"""Compute per-step velocity from consecutive rows: v = delta(position) / delta(time)."""
	velocity_rows: list[dict] = []

	for prev, curr in zip(rows, rows[1:]):
		dt = curr["time_s"] - prev["time_s"]
		if dt <= 0.0:
			continue

		vx = (curr["x"] - prev["x"]) / dt
		vy = (curr["y"] - prev["y"]) / dt
		vz = (curr["z"] - prev["z"]) / dt

		velocity_rows.append(
			{
				"prev_frame": prev["frame"],
				"frame": curr["frame"],
				"time_s": curr["time_s"],
				"dt": dt,
				"vx": vx,
				"vy": vy,
				"vz": vz,
				"magnitude": math.sqrt(vx * vx + vy * vy + vz * vz),
			}
		)

	return velocity_rows


def percentile(sorted_values: list[float], p: float) -> float:
	"""Return percentile using linear interpolation; input must already be sorted."""
	if not sorted_values:
		return 0.0
	if len(sorted_values) == 1:
		return sorted_values[0]

	idx = (len(sorted_values) - 1) * p
	low = int(math.floor(idx))
	high = int(math.ceil(idx))
	if low == high:
		return sorted_values[low]

	frac = idx - low
	return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def remove_high_magnitude_outliers(rows: list[dict], iqr_multiplier: float = 1.5) -> tuple[int, float | None]:
	"""Remove very high magnitude outliers in-place using an upper IQR fence."""
	if len(rows) < 4:
		return 0, None

	magnitudes = sorted(row["magnitude"] for row in rows)
	q1 = percentile(magnitudes, 0.25)
	q3 = percentile(magnitudes, 0.75)
	iqr = q3 - q1
	upper_fence = q3 + iqr_multiplier * iqr

	original_count = len(rows)
	rows[:] = [row for row in rows if row["magnitude"] <= upper_fence]
	removed_count = original_count - len(rows)
	return removed_count, upper_fence


def add_minmax_scaled_magnitude(rows: list[dict]) -> None:
	"""Add min-max scaled magnitude to each velocity row in-place."""
	if not rows:
		return

	magnitudes = [row["magnitude"] for row in rows]
	min_mag = min(magnitudes)
	max_mag = max(magnitudes)
	range_mag = max_mag - min_mag

	if range_mag == 0.0:
		for row in rows:
			row["magnitude_minmax"] = 0.0
		return

	for row in rows:
		row["magnitude_minmax"] = (row["magnitude"] - min_mag) / range_mag


def write_velocity_csv(rows: list[dict], csv_path: Path) -> None:
	fieldnames = ["prev_frame", "frame", "time_s", "dt", "vx", "vy", "vz", "magnitude", "magnitude_minmax"]
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def ask_hr_from_terminal() -> float:
	"""Prompt user for heart rate value in terminal."""
	while True:
		raw = input("Enter hr: ").strip()
		try:
			return float(raw)
		except ValueError:
			print("Invalid hr. Please enter a numeric value.")


def load_model(model_path: Path):
	"""Load the pickled sklearn model."""
	with model_path.open("rb") as f:
		return pickle.load(f)


def predict_model_output(model, model_inputs: dict[str, float]):
	"""Run model.predict with inputs ordered by feature names when available."""
	if hasattr(model, "feature_names_in_"):
		ordered_names = list(model.feature_names_in_)
		x_row = [model_inputs[name] for name in ordered_names]
	else:
		x_row = [
			model_inputs["hr"],
			model_inputs["velocity_magnitude_scaled"],
		]

	prediction = model.predict([x_row])
	return prediction[0]


def main() -> None:
	left_ankle_rows = load_left_ankle_rows(INPUT_CSV)
	velocity_rows = compute_velocity(left_ankle_rows)
	removed_outliers, outlier_upper_fence = remove_high_magnitude_outliers(velocity_rows)
	add_minmax_scaled_magnitude(velocity_rows)
	write_velocity_csv(velocity_rows, OUTPUT_CSV)

	hr = ask_hr_from_terminal()
	velocity_magnitude_scaled = velocity_rows[200]["magnitude_minmax"] if velocity_rows else 0.0
	model_inputs = {
		"hr": hr,
		"velocity_magnitude_scaled": velocity_magnitude_scaled,
	}
	model = load_model(MODEL_PKL)
	model_output = predict_model_output(model, model_inputs)

	print(f"Filtered {len(left_ankle_rows)} rows for {TARGET_LANDMARK}.")
	if outlier_upper_fence is not None:
		print(f"Removed {removed_outliers} high-magnitude outliers above {outlier_upper_fence:.6f}.")
	print(f"Wrote {len(velocity_rows)} velocity rows to {OUTPUT_CSV}.")
	print(f"Model input payload: {model_inputs}")
	print(f"Model output: {model_output}")


if __name__ == "__main__":
	main()
