from dataclasses import dataclass
import os
from typing import Optional, List


@dataclass
class Thresholds:
	"""Cosine-distance thresholds per model (lower is closer)."""
	arcface: float = 0.35
	facenet: float = 0.40
	vggface: float = 0.40


@dataclass
class AppConfig:
	"""Runtime configuration for the attendance system."""
	# Default student id for simple wrappers
	default_student_id: str = os.environ.get("DEFAULT_STUDENT_ID", "default")
	# Preferred metric (cosine distance). Cosine distance d = 1 - cosine_similarity.
	metric: str = "cosine"
	# Detection size for InsightFace detector
	det_size: tuple[int, int] = (640, 640)
	# Thresholds per model
	thresholds: Thresholds = Thresholds()
	# Sensitivity in [0, 1]; higher means stricter (lower allowable distance)
	sensitivity: float = 0.5
	# SQLite DB path
	db_path: str = os.environ.get(
		"ATTENDANCE_DB_PATH", os.path.join(os.path.dirname(__file__), "attendance.db")
	)


def adjust_thresholds_for_sensitivity(thresholds: Thresholds, sensitivity: float) -> Thresholds:
	"""Linearly scale thresholds around defaults using a sensitivity in [0, 1].

	Higher sensitivity -> lower threshold (stricter). We apply a ±20% window around defaults.
	"""
	sensitivity = max(0.0, min(1.0, sensitivity))
	# 0 -> +20% (looser), 1 -> -20% (stricter)
	factor = 1.2 - 0.4 * sensitivity
	return Thresholds(
		arcface=thresholds.arcface * factor,
		facenet=thresholds.facenet * factor,
		vggface=thresholds.vggface * factor,
	)


def get_ort_providers() -> List[str]:
	"""Return best available onnxruntime providers (CUDA preferred)."""
	try:
		import onnxruntime as ort
		available = ort.get_available_providers()
		if "CUDAExecutionProvider" in available:
			return ["CUDAExecutionProvider", "CPUExecutionProvider"]
		return ["CPUExecutionProvider"]
	except Exception:
		return ["CPUExecutionProvider"]