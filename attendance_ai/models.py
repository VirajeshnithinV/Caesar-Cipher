from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np

from .config import AppConfig, get_ort_providers


class ArcFaceModel:
	"""ArcFace embedding extractor via InsightFace FaceAnalysis.

	Uses onnxruntime providers with CUDA when available.
	"""

	_lock = threading.Lock()
	_instance: Optional["ArcFaceModel"] = None

	def __init__(self, config: AppConfig):
		from insightface.app import FaceAnalysis

		self.config = config
		self.providers = get_ort_providers()
		self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
		# ctx_id 0 uses first GPU if CUDA provider is selected
		ctx_id = 0 if self.providers and self.providers[0] == "CUDAExecutionProvider" else -1
		self.app.prepare(ctx_id=ctx_id, det_size=self.config.det_size)

	@classmethod
	def get(cls, config: AppConfig) -> "ArcFaceModel":
		with cls._lock:
			if cls._instance is None:
				cls._instance = ArcFaceModel(config)
			return cls._instance

	def get_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
		faces = self.app.get(image_bgr)
		if not faces:
			return None
		# Pick largest face by bounding box area
		largest = max(
			faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
		)
		# normed_embedding is already L2-normalized
		emb = largest.normed_embedding
		return np.asarray(emb, dtype=np.float32)


class DeepFaceModel:
	"""DeepFace embedding extractor for a given model name (e.g., Facenet, VGG-Face)."""

	_instances: Dict[str, "DeepFaceModel"] = {}
	_lock = threading.Lock()

	def __init__(self, model_name: str):
		from deepface import DeepFace

		self.model_name = model_name
		self.model = DeepFace.build_model(model_name)

	@classmethod
	def get(cls, model_name: str) -> "DeepFaceModel":
		with cls._lock:
			if model_name not in cls._instances:
				cls._instances[model_name] = DeepFaceModel(model_name)
			return cls._instances[model_name]

	def get_embedding(self, image_bgr_or_rgb: np.ndarray) -> Optional[np.ndarray]:
		from deepface import DeepFace

		reps = DeepFace.represent(
			img_path=image_bgr_or_rgb,
			model_name=self.model_name,
			model=self.model,
			detector_backend="retinaface",
			enforce_detection=False,
		)
		if not reps:
			return None
		feat = reps[0]["embedding"]
		return np.asarray(feat, dtype=np.float32)


class ModelRegistry:
	"""Singleton-style registry to access all three models."""

	_lock = threading.Lock()
	_instance: Optional["ModelRegistry"] = None

	def __init__(self, config: AppConfig):
		self.config = config
		self.arcface = ArcFaceModel.get(config)
		self.facenet = DeepFaceModel.get("Facenet")
		self.vggface = DeepFaceModel.get("VGG-Face")

	@classmethod
	def get(cls, config: AppConfig) -> "ModelRegistry":
		with cls._lock:
			if cls._instance is None:
				cls._instance = ModelRegistry(config)
			return cls._instance

	def extract_all(self, image_bgr: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
		# DeepFace accepts BGR/RGB numpy arrays; internally handles conversion.
		return {
			"arcface": self.arcface.get_embedding(image_bgr),
			"facenet": self.facenet.get_embedding(image_bgr),
			"vggface": self.vggface.get_embedding(image_bgr),
		}