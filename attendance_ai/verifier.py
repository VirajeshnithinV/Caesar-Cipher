from typing import Dict, Optional, Tuple

import numpy as np

from .config import AppConfig, Thresholds
from .utils.metrics import compute_distance, distance_to_confidence, l2_normalize


def compare_embeddings(
	config: AppConfig,
	thresholds: Thresholds,
	stored: Dict[str, Optional[np.ndarray]],
	query: Dict[str, Optional[np.ndarray]],
) -> Tuple[bool, Dict[str, float], Dict[str, bool], float, int]:
	"""Compare stored vs query embeddings across models and majority vote.

	Returns: (is_match, distances, passes, confidence, votes)
	"""
	distances: Dict[str, float] = {}
	passes: Dict[str, bool] = {}
	confidences: Dict[str, float] = {}

	available_models = [
		("arcface", thresholds.arcface),
		("facenet", thresholds.facenet),
		("vggface", thresholds.vggface),
	]

	valid_models = 0
	vote_count = 0
	for name, thr in available_models:
		ref = stored.get(name)
		q = query.get(name)
		if ref is None or q is None:
			continue
		# Normalize both for fair cosine comparison
		ref_n = l2_normalize(ref.astype(np.float32))
		q_n = l2_normalize(q.astype(np.float32))
		d = compute_distance(config.metric, ref_n, q_n)
		distances[name] = d
		is_pass = d <= thr
		passes[name] = is_pass
		confidences[name] = distance_to_confidence(d, thr)
		valid_models += 1
		if is_pass:
			vote_count += 1

	# Decide by majority
	is_match = vote_count >= 2
	# Aggregate confidence: average of available model confidences
	final_conf = float(np.mean(list(confidences.values()))) if confidences else 0.0
	return is_match, distances, passes, final_conf, vote_count