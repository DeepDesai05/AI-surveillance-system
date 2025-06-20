# 🛡️ AI-Based Smart Surveillance System

An advanced Computer Vision project that detects weapons, intrusions, and suspicious activity using YOLOv8 and Vision-Language Models.

---

## 🔍 Features

- 🔫 **Weapon detection** (guns, knives)
- 🧗‍♂️ **Wall jump detection**
- 🚫 **Restricted area intrusion**
- 🤖 **VLM integration** for understanding scenes
- 🧠 Trained on a **custom dataset** with YOLOv8
- 📷 Plans to **alert authorities** with snapshots

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, PyTorch, Ultralytics YOLOv8, NumPy
- **Model:** YOLOv8 (custom trained)
- **Optional Integration:** VLM (Vision-Language Models)

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `CV.py` | Core computer vision logic (YOLOv8) |
| `CAM.py` | Camera input module |
| `VLM.py` | Vision-Language Model integration |
| `best (6).pt` | Trained YOLOv8 model weights |
| `data.yaml` | YOLO class labels and paths |

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/DeepDesai05/ai-surveillance-system.git
cd ai-surveillance-system

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics opencv-python torch numpy
python CV.py

