## Attendance AI - Face Verification Attendance System

This package provides a production-oriented, GPU-accelerated face verification system for automated attendance.
It compares a student's selfie with their stored enrollment photo using three SOTA models (ArcFace, FaceNet, VGG-Face),
performs majority voting, and returns a clear decision with confidence.

### Features
- ArcFace (InsightFace + onnxruntime-gpu), FaceNet, VGG-Face (DeepFace)
- Embedding-based verification with cosine distance
- Majority voting across the three models to reduce false positives
- SQLite storage for per-student embeddings
- CUDA acceleration for ArcFace via onnxruntime-gpu
- Tunable thresholds per model and global sensitivity adjustments

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- ArcFace runs on GPU via `onnxruntime-gpu` if CUDA is available; otherwise it falls back to CPU.
- DeepFace will use TensorFlow under the hood; GPU setup for TensorFlow is optional and not required for correctness.

### Quickstart
```python
from attendance_ai.api import register_student, verify_student, AttendanceSystem

# Option A: use default student id (env DEFAULT_STUDENT_ID or "default")
register_student("/path/to/enrollment_photo.jpg")
result = verify_student("/path/to/live_selfie.jpg")
print(result)  # {"result": "Same Person" | "Not Same Person", "confidence": 0.0..1.0}

# Option B: multi-student usage via the class API
a = AttendanceSystem()
a.register_student("student_123", "/path/to/student_123_enroll.jpg")
print(a.verify_student("student_123", "/path/to/student_123_selfie.jpg"))
```

### Threshold Tuning
- Default cosine-distance thresholds (lower is closer) are set conservatively.
- You can adjust per-model thresholds or set a global sensitivity:
```python
from attendance_ai.config import Thresholds
from attendance_ai.api import AttendanceSystem

system = AttendanceSystem()
system.update_thresholds(Thresholds(arcface=0.35, facenet=0.40, vggface=0.45))
# Or a convenience sensitivity in [0, 1] where higher is stricter
after = system.set_sensitivity(0.7)
```

### Storage
- A local SQLite DB is created at `attendance_ai/attendance.db`.
- Embeddings are stored per student per model, enabling fast repeated verification.

### Notes
- For best results, enroll high-quality, frontal, well-lit photos.
- Selfies should include the full face with minimal occlusions.
- If you use your own detection/ROI cropping, ensure consistent alignment across enrollment and verification.

### License
MIT