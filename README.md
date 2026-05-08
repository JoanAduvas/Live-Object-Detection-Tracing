A real-time object detection application built with *Python*, *YOLOv8*, and *Streamlit*. This app features live webcam detection, automated logging, and a *Twilio SMS alert system* that notifies you whenever a specific target object is identified.

## ✨ Features

- *Real-time Detection:* Powered by YOLOv8 (Nano) for high-speed and accurate inference.
- *WebRTC Integration:* Seamless low-latency video streaming in the browser via streamlit-webrtc.
- *Smart Alerts:* Select a target object (e.g., "person", "dog") and receive an SMS notification via Twilio.
- *Automated Logging:* Captures and saves screenshots of new detections to the detection_logs/ directory.
- *Detection Gallery:* A built-in management interface to view, download (as ZIP), or delete captured images.
- *Spam Protection:* Includes a 60-second cooldown timer for SMS alerts to prevent excessive messaging.

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [[https://github.com/YOUR_USERNAME/live-object-detection-tracing.git](https://github.com/YOUR_USERNAME/live-object-detection-tracing.git)](https://github.com/JoanAduvas/Live-Object-Detection-Tracing.git)
cd live-object-detection-tracing
2. Install Dependencies
Ensure you have Python installed, then run:

Bash
pip install -r requirements.txt
3. System Requirements (Linux/Cloud)
For deployment on Streamlit Cloud, ensure your packages.txt contains:

📂 Project Structure
app.py - The main application logic.

requirements.txt - Python library dependencies.

packages.txt - System-level dependencies for Linux environments.

detection_logs/ - Directory for saved detection images (auto-generated).

👨‍💻 Developer
Joan M. Aduvas

Computer Science Student | Dr. Emilio B. Espinosa Sr. Memorial State College of Agriculture and Technology

Disclaimer: This project is for educational and research purposes in the field of Computer Vision.
