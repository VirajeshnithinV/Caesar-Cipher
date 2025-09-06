import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EmbeddingRecord:
	student_id: str
	embeddings: Dict[str, np.ndarray]  # keys: arcface, facenet, vggface


class EmbeddingStore:
	"""SQLite-backed storage for student embeddings.

	Stores numpy arrays as serialized blobs via np.save / np.load.
	"""

	def __init__(self, db_path: str):
		self.db_path = db_path
		os.makedirs(os.path.dirname(db_path), exist_ok=True)
		self._conn = sqlite3.connect(self.db_path)
		self._conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS embeddings (
				student_id TEXT PRIMARY KEY,
				arcface BLOB,
				facenet BLOB,
				vggface BLOB,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)
		self._conn.commit()

	def close(self) -> None:
		try:
			self._conn.close()
		except Exception:
			pass

	@staticmethod
	def _np_to_blob(arr: Optional[np.ndarray]) -> Optional[bytes]:
		if arr is None:
			return None
		import io

		buf = io.BytesIO()
		np.save(buf, arr)
		return buf.getvalue()

	@staticmethod
	def _blob_to_np(blob: Optional[bytes]) -> Optional[np.ndarray]:
		if blob is None:
			return None
		import io

		buf = io.BytesIO(blob)
		buf.seek(0)
		return np.load(buf, allow_pickle=False)

	def upsert_embeddings(self, student_id: str, embeddings: Dict[str, Optional[np.ndarray]]) -> None:
		arc_blob = self._np_to_blob(embeddings.get("arcface"))
		fac_blob = self._np_to_blob(embeddings.get("facenet"))
		vgg_blob = self._np_to_blob(embeddings.get("vggface"))
		self._conn.execute(
			"""
			INSERT INTO embeddings (student_id, arcface, facenet, vggface)
			VALUES (?, ?, ?, ?)
			ON CONFLICT(student_id) DO UPDATE SET
				arcface = excluded.arcface,
				facenet = excluded.facenet,
				vggface = excluded.vggface,
				updated_at = CURRENT_TIMESTAMP
			""",
			(student_id, arc_blob, fac_blob, vgg_blob),
		)
		self._conn.commit()

	def get_embeddings(self, student_id: str) -> Optional[EmbeddingRecord]:
		cur = self._conn.execute(
			"SELECT arcface, facenet, vggface FROM embeddings WHERE student_id = ?",
			(student_id,),
		)
		row = cur.fetchone()
		if not row:
			return None
		arc, fac, vgg = row
		return EmbeddingRecord(
			student_id=student_id,
			embeddings={
				"arcface": self._blob_to_np(arc),
				"facenet": self._blob_to_np(fac),
				"vggface": self._blob_to_np(vgg),
			},
		)