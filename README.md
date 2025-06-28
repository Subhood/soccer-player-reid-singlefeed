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

### 🔗 Model Download (Required Before Running)

Please download the YOLOv11 model from the link below and place it inside the `models/` folder:

📥 [Download yolov11_player.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

📁 Destination: `models/yolov11_player.pt`

---

## 🗂️ Directory Structure

<pre> soccer-player-reid-singlefeed/ 
   ├── models/ 
   │ └── yolov11_player.pt ← ⬇️ Download manually & place here 
   ├── videos/ 
   │ └── 15sec_input_720p.mp4 
   ├── outputs/ 
   │ ├── output_20250625_220045.mp4 
   │ └── log_20250625_220045.csv 
   ├── src/ 
   │ ├── detect.py 
   │ ├── reid.py 
   │ ├── track.py 
   │ └── utils.py 
   ├── main.py 
   ├── requirements.txt 
   ├── report.pdf 
   ├── README.md 
   └── .gitignore </pre>

---

## 📦 Installation

1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows

## Install dependencies:
pip install -r requirements.txt

## Run the pipeline:
python main.py

## 👨‍💻 Author

**Name**: Subhood M  
**Submission Option**: Option 2 – Single Feed  
**Email**: Subhood2004@gmail.com
