# NrityaVision AI
Intelligent Kuchipudi hand-mudra recognition and performance evaluation (Python, OpenCV, MediaPipe, scikit-learn, DTW, Tkinter, SQLite, Matplotlib, ReportLab).

## 1. Install (Windows)
1. Install **64-bit Python 3.10, 3.11 or 3.12** from python.org (tick "Add python.exe to PATH" and keep tcl/tk).
2. Open this folder in VS Code (or CMD) and run **`setup_windows.bat`** - it creates `.venv`, installs `requirements.txt` and runs the self-test.
   Manual alternative:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python tools\selftest.py
   ```
3. Start: **`run.bat`** or `.venv\Scripts\python.exe main.py`.

## 2. First-time workflow (the model starts EMPTY - nothing is pre-trained)
No public labelled Kuchipudi mudra dataset ships with this project, so you must create one:
1. **Reference Mudra Library > Dataset & Training**: select a mudra, press **Capture from webcam** (3 s countdown, then samples are taken automatically; slowly turn/tilt the hand and vary distance). Or import images / a folder of images / a video clip.
2. Repeat for all five mudras, ideally in **several separate sessions** (different days, lighting, people, left and right hand). Aim for 100+ samples per mudra, minimum 10.
3. Press **Train model**. Three model families are compared by cross-validation, the best is kept. Validation is grouped by capture session when possible; if the log says *optimistic*, record more sessions.
4. Have your guru check the reference images - similarity can only be as correct as the references.
5. Use **Live Camera**, **Upload Image**, **Upload Dance Video**; reports are in **Performance Reports** (PDF/CSV export).

## 3. How results are computed (no random numbers)
* MediaPipe gives 21 landmarks per hand. They are centred on the wrist, scaled by palm size, left hands mirrored, then converted to rotation-invariant features (joint angles, fingertip distances, finger spread, palm orientation).
* **Confidence** = classifier probability. A hand is reported as **"Unable to recognize"** when it is hidden, cut off, blurry, too dark/bright, too small, the confidence or margin is low, or the shape is unlike every trained mudra. Frames with more than two hands (several dancers) are skipped by default.
* **Similarity (0-100)** = geometric closeness to the mean of your labelled reference samples with per-feature tolerance (formula in `nrityavision/similarity.py`). Confidence is never shown as correctness.
* Status: similarity >= 75 Correct, 55-75 Needs improvement, < 55 Incorrect (changeable in Settings). Wrong or missed mudra scores 0. **Overall accuracy = mean item score.**
* Video: sampled at ~10 fps -> per-frame recognition -> 5-frame majority smoothing -> segments (holds shorter than 0.4 s are transitions) -> evaluation. With an **expected sequence** (typed or a reference video) segments are aligned with **DTW**; skipped/extra mudras are flagged. With a **reference video**, hand-shape trajectories are also compared by DTW (movement similarity + deviation timestamps, speed differences ignored).

## 4. Troubleshooting
| Problem | Fix |
|---|---|
| `import cv2` fails / numpy 2 error | Do **not** install `opencv-python` next to MediaPipe: `pip uninstall -y opencv-python opencv-contrib-python` then `pip install -r requirements.txt` |
| `module 'mediapipe' has no attribute 'solutions'` | Newer MediaPipe removed the legacy API: `pip install mediapipe==0.10.14` |
| No matching distribution for mediapipe | Use 64-bit Python 3.10-3.12 |
| Camera does not open / black | Close Zoom/Teams, Windows Settings > Privacy > Camera, change *Camera index* in Settings |
| Many "blurry" / "too dark" rejections | Better light, steadier camera, or lower *Blur threshold* / *Minimum brightness* in Settings |
| Left/right hands swapped in uploaded media | Toggle *Uploads are mirrored* in Settings |
| Video will not open | Convert to H.264 MP4 (HandBrake/VLC) |
| Everything "Unable to recognize" | Model not trained or too few/varied samples; lower *Minimum confidence* only after collecting more data |
| Error details | `data/logs/app.log` |

## 5. Limitations
* Accuracy depends entirely on the labelled data you collect; check the cross-validation report before trusting results.
* Similar mudras (e.g. Tripataka vs Mayura) need many varied samples. 2D webcam landmarks have depth ambiguity; a side-on hand can be unreliable.
* Descriptions of mudras vary between schools; the app compares with *your* references. It supports learning, it does not replace a teacher.
* Only hand mudras are analysed (no body/face/rhythm yet).

## 6. Layout
`main.py` entry | `nrityavision/` core (`features`, `hand_tracker`, `classifier`, `similarity`, `dtw`, `analyzer`, `report`, `database`) and `ui/` pages | `tools/selftest.py` | `data/` (SQLite DB, reference media, reports, logs) | `models/` trained model.
Add the other Asamyuta Hastas via *Add all 28 Asamyuta Hastas*, then collect samples and retrain - no code changes needed.
