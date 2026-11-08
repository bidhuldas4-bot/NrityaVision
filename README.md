# NrityaVision AI

## AI-Powered Kuchipudi Mudra Recognition and Performance Evaluation

NrityaVision is an AI-powered classical dance Mudra recognition and analysis system designed to help dancers learn, practice, and improve their hand gestures. It uses Python, OpenCV, MediaPipe, and Machine Learning to detect and evaluate hand Mudras through webcam input, uploaded images, and dance videos.

Users can upload reference Mudra images, collect custom training data, train recognition models, and evaluate the accuracy of their hand gestures. The system also supports dance video analysis to identify Mudras and provide performance feedback.

**Technologies:** Python, OpenCV, MediaPipe, Scikit-learn, DTW, Tkinter, SQLite, NumPy, Matplotlib, ReportLab.

---

## 1. Key Features

- Real-Time Mudra Detection using a webcam.
- Upload and analyze hand gesture images.
- Custom Mudra dataset collection and model training.
- Pataka Mudra training and evaluation.
- Dance video analysis.
- Mudra recognition and similarity scoring.
- Performance report generation in PDF and CSV formats.
- Reference Mudra Library.
- Database management using SQLite.

---

## 2. Installation (Windows)

### Requirements

- Windows 10 or Windows 11.
- 64-bit Python 3.10, 3.11, or 3.12.
- Visual Studio Code.
- Webcam for real-time detection.

### Installation Steps

1. Download Python from https://www.python.org/downloads/.
2. Enable **Add Python to PATH** during installation.
3. Open the NrityaVision project folder in VS Code.
4. Open the VS Code terminal.
5. Create a virtual environment:

```bash
python -m venv .venv
```

6. Activate the environment:

```powershell
.venv\Scripts\activate
```

7. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

8. Install dependencies:

```bash
pip install -r requirements.txt
```

9. Run the self-test:

```bash
python tools/selftest.py
```

### Start the Application

```bash
python main.py
```

Alternatively, run:

```bash
run.bat
```

---

## 3. Mudra Dataset Collection and Model Training

The recognition model starts empty. Users must collect and label their own Mudra training data.

### Step 1: Open Dataset & Training

Navigate to:

**Reference Mudra Library → Dataset & Training**

### Step 2: Collect Training Images

Users can collect training samples using:

- Webcam capture.
- Individual image uploads.
- Folder-based image imports.
- Video clip imports.

### Step 3: Organize the Dataset

Example dataset structure:

```text
dataset/
│
├── Pataka/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── Tripataka/
│   ├── image1.jpg
│   └── image2.jpg
│
└── Mayura/
    ├── image1.jpg
    └── image2.jpg
```

Collect samples with different hand angles, lighting conditions, distances, and backgrounds.

Aim for 100 or more samples per Mudra where possible. A minimum of 10 samples can be used for initial experimentation, but more data is needed for reliable recognition.

### Step 4: Train the Model

1. Open Dataset & Training.
2. Select the Mudra dataset.
3. Click **Train Model**.
4. Wait for training and validation.
5. Review the evaluation results.

### Step 5: Test the Model

Test the trained model using:

- Live Camera.
- Upload Image.
- Upload Dance Video.

---

## 4. How Mudra Recognition Works

NrityaVision uses MediaPipe Hands to detect 21 hand landmarks.

The recognition process includes:

1. Hand Detection.
2. Hand Landmark Extraction.
3. Feature Extraction and Normalization.
4. Machine Learning Classification.
5. Mudra Similarity Calculation.
6. Accuracy Evaluation.
7. Performance Report Generation.

### Accuracy Evaluation

The system compares detected hand positions with labelled reference samples.

Similarity scores are measured from 0 to 100.

| Similarity Score | Result |
|---|---|
| 75–100 | Correct |
| 55–74 | Needs Improvement |
| Below 55 | Incorrect |

These thresholds can be adjusted in Settings.

The system may display **Unable to Recognize** when the hand is hidden, blurry, too small, or poorly detected.

Classification confidence and geometric similarity are separate measurements.

---

## 5. Dance Video Analysis

NrityaVision supports uploaded dance video analysis.

The video analysis process includes:

1. Upload a dance video.
2. Extract frames from the video.
3. Detect hand landmarks.
4. Recognize Mudras in individual frames.
5. Smooth recognition results across frames.
6. Identify Mudra sequences.
7. Evaluate gesture accuracy.
8. Generate a performance report.

When an expected Mudra sequence or reference video is provided, Dynamic Time Warping (DTW) can be used to compare sequences and movement patterns.

---

## 6. Supported Mudras

The system supports custom Mudra training.

Users can start with Pataka Mudra and gradually add other Indian classical dance Mudras.

Additional Mudras require labelled training samples before reliable recognition is possible.

The project also includes an option to add all 28 Asamyuta Hastas through the Reference Mudra Library, if enabled in the application.

---

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| MediaPipe | Hand landmark detection |
| Scikit-learn | Machine Learning |
| NumPy | Numerical calculations |
| DTW | Sequence comparison |
| Tkinter | Desktop user interface |
| SQLite | Database management |
| Matplotlib | Data visualization |
| ReportLab | PDF report generation |

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| Python package installation error | Check Python version and install dependencies again. |
| MediaPipe installation error | Use compatible 64-bit Python 3.10–3.12. |
| Camera does not open | Check camera permissions and close other applications using the camera. |
| Black camera screen | Check the camera index in Settings. |
| Model cannot recognize Mudras | Collect more varied training samples and retrain. |
| Poor recognition accuracy | Improve dataset quality and collect more samples. |
| Video cannot open | Convert the video to H.264 MP4 format. |
| Application startup error | Check data/logs/app.log for error details. |

---

## 9. Project Structure

```text
NrityaVision/
│
├── main.py
├── requirements.txt
├── README.md
├── setup_windows.bat
├── run.bat
│
├── nrityavision/
│   ├── features/
│   ├── hand_tracker/
│   ├── classifier/
│   ├── similarity/
│   ├── dtw/
│   ├── analyzer/
│   ├── report/
│   └── database/
│
├── ui/
│
├── tools/
│   └── selftest.py
│
├── data/
│   ├── reference_media/
│   ├── reports/
│   └── logs/
│
└── models/
```

The actual folder structure may vary depending on the project version.

---

## 10. Limitations

- Recognition accuracy depends on the quality and quantity of training data.
- Similar Mudras may require more training samples.
- Webcam-based 2D hand detection can have depth and orientation limitations.
- Different dance schools may follow different Mudra variations.
- The application focuses on hand Mudra recognition, not complete body posture, facial expressions, or rhythm analysis.
- AI-generated evaluations are intended as learning assistance and do not replace guidance from a qualified dance teacher.

---

## 11. Future Enhancements

- Support for more Indian classical dance Mudras.
- Improved real-time Mudra recognition.
- Advanced hand movement analysis.
- Better dance video performance evaluation.
- Improved AI-based feedback.
- Mobile application support.
- More accurate reference-based dance evaluation.

---

## 12. Project Information

**Project Name:** NrityaVision AI

**Domain:** Artificial Intelligence and Machine Learning

**Project Type:** AI-Based Classical Dance Gesture Recognition and Analysis

**Programming Language:** Python

**Platform:** Windows Desktop Application

---

## 13. Author

**Developed By:** Bidhul Das

---

## 14. License

This project is developed for educational and learning purposes. A suitable open-source license can be added if the project is distributed publicly.

---

**NrityaVision AI – Bringing Artificial Intelligence into Indian Classical Dance.**