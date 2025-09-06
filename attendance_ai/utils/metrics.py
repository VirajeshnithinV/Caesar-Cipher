from typing import Literal, Optional

import numpy as np


def l2_normalize(emb: np.ndarray, eps: float = 1e-10) -> np.ndarray:
	norm = np.linalg.norm(emb, ord=2, axis=-1, keepdims=True)
	return emb / (norm + eps)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
	a = l2_normalize(a, eps)
	b = l2_normalize(b, eps)
	cos_sim = float(np.dot(a, b))
	return 1.0 - cos_sim


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.linalg.norm(a - b))


def compute_distance(metric: Literal["cosine", "euclidean"], a: np.ndarray, b: np.ndarray) -> float:
	if metric == "cosine":
		return cosine_distance(a, b)
	elif metric == "euclidean":
		return euclidean_distance(a, b)
	else:
		raise ValueError(f"Unsupported metric: {metric}")


def distance_to_confidence(distance: float, threshold: float) -> float:
	"""Map a distance to [0, 1] confidence given a threshold.

	Distances below threshold yield higher confidence, clipped to [0, 1].
	"""
	if threshold <= 0:
		return 0.0
	score = (threshold - distance) / threshold
	return float(max(0.0, min(1.0, score)))