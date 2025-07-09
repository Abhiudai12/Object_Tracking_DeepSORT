# ⚽ Advanced Football Player Tracking using YOLOv11 + Deep SORT + EasyOCR

This project implements an AI-powered player tracking system that detects and re-identifies football players in a video using their jersey numbers. It was submitted as an assignment for the **Stealth Mode AI/ML Internship**.

---

## 🎯 Objective

To build a robust video analytics pipeline that:
- Detects football players using a custom-trained YOLOv11 model.
- Tracks each player across frames using **Deep SORT**.
- Recognizes jersey numbers using **EasyOCR**.
- Maintains consistent identity for each player throughout the video.

---

##  Technologies Used

| Component         | Tool/Library                         |
|------------------|--------------------------------------|
| **Object Detection** | [fine-tuned YOLOv11 for player-ball detection](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view) |
| **Object Tracking**  | [Deep SORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/master/deep_sort/deep_sort) |
| **OCR**              | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| **Progress Bar**     | [`tqdm`](https://tqdm.github.io/) |
| **Video Processing** | OpenCV |
| **Device**           | GPU/CPU (PyTorch auto-detect) |
| **Utilities**        | NumPy, defaultdict, deque, gdown |

---

## 🖼️ Output Example

📹 **Final Output Video**  
▶️ [Click to watch output video](./output_tracking.mp4)

This video shows:
- Bounding boxes on detected players.
- Real-time track IDs of players.
- Smooth consistent tracking with OCR-based identification.

---

## 📊 Player Jersey Number Mapping (Summary)

| Track ID | Jersey Number |
|----------|----------------|
| 13       | 28             |
| 45       | 10             |
| 50       | 13             |
| 55       | 52             |
| 52       | 6              |
| 105      | 5              |
| 141      | 08             |
| 149      | 10             |
| 87       | 10             |
| 168      | 2              |
| 159      | 57             |

> IDs are maintained consistently even with camera motion and overlapping players.

---

## 🧪 Improvements & Learnings

- * Reduced OCR frequency using a 15-frame gap per track to avoid jitter.
- * Ignored unreasonably fast jumps to filter ID switching.
- * Focused OCR only on torso regions to reduce noise.
- * Some jersey readings may repeat due to occlusion or blur.

---

##  Setup Instructions

```bash
# 1. Install dependencies
pip install ultralytics deep_sort_realtime easyocr opencv-python tqdm filterpy gdown

# 2. Run the script
python player_tracking.py
