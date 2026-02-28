# 🧑‍🦯 Vision Safe  
## AI-Based Indoor Safety & Navigation Assistant

Vision Safe is an AI-powered indoor safety and navigation assistant designed to enhance independence and mobility for visually impaired individuals. The system proactively detects environmental hazards in real time using computer vision and delivers prioritized voice guidance.

Unlike traditional mobility tools that detect obstacles only upon physical contact, Vision Safe predicts risks before collision using bounding-box-based proximity estimation and spatial awareness logic.


## 🚨 Problem Statement

Indoor navigation presents major challenges for visually impaired individuals because:

- GPS does not function effectively indoors  
- White canes detect obstacles only upon contact  
- There is no predictive awareness of environmental hazards  
- Crowded and low-visibility environments increase risk  

There is a need for a real-time assistive system capable of proactively detecting hazards and delivering actionable guidance.


## 💡 Proposed Solution

Vision Safe integrates:

- Real-time object detection (YOLOv8)  
- Proximity estimation using bounding-box scaling  
- Directional guidance (left / right / ahead)  
- Crowd density detection  
- Low-visibility detection  
- Priority-based hazard ranking  
- State-aware voice feedback  

The system follows a hybrid edge-based architecture:

- Inference runs locally for low latency  
- Backend API is containerized using Docker for deployment portability  


## 🏗️ System Architecture

Camera  
→ YOLOv8 Detection  
→ Proximity Estimation  
→ Direction Analysis  
→ Environmental Risk Detection  
→ Priority-Based Safety Engine  
→ Voice Feedback  
→ FastAPI Backend (Dockerized)


## 🛠️ Technology Stack

### AI & Vision
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  

### Backend
- FastAPI  
- Uvicorn  
- Python Threading  

### Assistive Output
- pyttsx3 (Offline Text-to-Speech)  

### Deployment
- Docker (Containerized API)  


## 📂 Project Structure

vision-safe/
│
├── backend/
│   ├── app.py
│   ├── detection/
│   │   └── detect_objects.py
│   ├── safety/
│   │   ├── obstacle_check.py
│   │   └── hazard_check.py
│   ├── guidance/
│   │   └── direction_helper.py
│   ├── voice/
│   │   └── speaker.py
│   ├── utils/
│   │   └── distance_estimator.py
│   └── models/
│       └── yolov8n.pt
│
├── frontend/
├── Dockerfile
├── requirements.txt
└── README.md


## ⚙️ Installation (Local Setup)

### 1. Clone Repository

git clone <your-repo-url>  
cd vision-safe  

### 2. Create Virtual Environment

python -m venv venv  
venv\Scripts\activate  (Windows)

### 3. Install Dependencies

pip install -r requirements.txt  

### 4. Run Locally

uvicorn backend.app:app --reload  

Open in browser:  
http://localhost:8000/docs  


## 🐳 Run Using Docker

### Build Image

docker build -t vision-safe .  

### Run Container

docker run -p 8000:8000 vision-safe  

Open in browser:  
http://localhost:8000/docs  

Note: Webcam access inside Docker on Windows may be limited due to hardware virtualization constraints. For full real-time inference, run locally.


## 🔄 System Workflow

1. Capture live camera frame  
2. Perform YOLOv8 object detection  
3. Estimate proximity using bounding-box area  
4. Determine direction via spatial segmentation  
5. Detect crowd density and low-visibility conditions  
6. Rank hazards using priority logic  
7. Deliver non-repetitive voice guidance  


## 🚀 Key Features

- Predictive obstacle detection  
- Directional spatial guidance  
- Crowd awareness  
- Low-visibility detection  
- Priority-based safety engine  
- Real-time voice alerts  
- Dockerized backend deployment  


## 🔮 Future Scope

- Stereo vision or depth-based distance estimation  
- Model optimization (ONNX / TensorRT)  
- Mobile or wearable integration  
- Multilingual voice support  
- Cloud-based safety analytics  


## 🏁 Conclusion

Vision Safe transforms reactive mobility into proactive safety by combining real-time computer vision, environmental risk assessment, and intelligent prioritization. The hybrid edge-based architecture ensures low-latency performance while maintaining deployment readiness through Docker.


