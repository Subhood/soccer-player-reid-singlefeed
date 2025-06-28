# ⚽ Soccer Player Re-Identification (Single Camera Feed)

This project implements a complete solution for re-identifying soccer players in a single 15-second video. Each player is assigned a consistent ID across frames — even after temporarily leaving and re-entering the camera view.

---

## 🎯 Objective

- Detect all players in a 15-second soccer video.
- Assign unique and consistent IDs.
- Re-identify players if they disappear and reappear later.

---

## 🛠️ Tools & Technologies

- **YOLOv11** (provided model for detection)
- **DeepSORT** (tracking across frames)
- **Color histogram features** (for re-identification)
- Python, OpenCV, NumPy, Scikit-learn

---

## 🗂️ Directory Structure

soccer-reid/
├── models/ # YOLOv11 model file (.pt)
├── videos/ # Input video file
├── outputs/ # Output video and logs
├── src/ # Core source code modules
│ ├── detect.py # Player detection
│ ├── track.py # DeepSORT tracking
│ ├── reid.py # Re-identification logic
│ └── utils.py # Visualization & logging
├── main.py # Pipeline entry point
├── requirements.txt # Python dependencies
├── README.md # This file
└── report.pdf # Final report 


---

## 📦 Installation

1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
