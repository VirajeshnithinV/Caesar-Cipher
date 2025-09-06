from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Union

import numpy as np

from .config import AppConfig, Thresholds, adjust_thresholds_for_sensitivity
from .db import EmbeddingStore
from .models import ModelRegistry
from .utils.images import load_image_bgr
from .verifier import compare_embeddings


class AttendanceSystem:
	def __init__(self, config: Optional[AppConfig] = None):
		self.config = config or AppConfig()
		self._registry = ModelRegistry.get(self.config)
		self._store = EmbeddingStore(self.config.db_path)

	def close(self) -> None:
		self._store.close()

	def update_thresholds(self, thresholds: Thresholds) -> Thresholds:
		self.config.thresholds = thresholds
		return thresholds

	def set_sensitivity(self, sensitivity: float) -> Thresholds:
		self.config.sensitivity = max(0.0, min(1.0, sensitivity))
		self.config.thresholds = adjust_thresholds_for_sensitivity(
			self.config.thresholds, self.config.sensitivity
		)
		return self.config.thresholds

	def register_student(self, student_id: str, photo: Union[str, np.ndarray]) -> Dict[str, bool]:
		image_bgr = load_image_bgr(photo)
		embs = self._registry.extract_all(image_bgr)
		# require at least two embeddings for robust verification
		num_present = sum(1 for v in embs.values() if v is not None)
		if num_present < 2:
			raise ValueError("Could not detect a face reliably in enrollment photo.")
		self._store.upsert_embeddings(student_id, embs)
		return {k: v is not None for k, v in embs.items()}

	def verify_student(self, student_id: str, selfie: Union[str, np.ndarray]) -> Dict[str, Union[str, float]]:
		stored = self._store.get_embeddings(student_id)
		if stored is None:
			raise ValueError(f"No enrollment found for student_id={student_id}")
		image_bgr = load_image_bgr(selfie)
		query_embs = self._registry.extract_all(image_bgr)
		thresholds = self.config.thresholds
		is_match, distances, passes, confidence, votes = compare_embeddings(
			self.config, thresholds, stored.embeddings, query_embs
		)
		result = "Same Person" if is_match else "Not Same Person"
		return {"result": result, "confidence": float(confidence)}


# Convenience top-level wrappers for single-student usage
_system_singleton: Optional[AttendanceSystem] = None


def _get_system_singleton() -> AttendanceSystem:
	global _system_singleton
	if _system_singleton is None:
		_system_singleton = AttendanceSystem()
	return _system_singleton


def register_student(photo: Union[str, np.ndarray]) -> Dict[str, bool]:
	sys = _get_system_singleton()
	return sys.register_student(sys.config.default_student_id, photo)


def verify_student(selfie: Union[str, np.ndarray]) -> Dict[str, Union[str, float]]:
	sys = _get_system_singleton()
	return sys.verify_student(sys.config.default_student_id, selfie)