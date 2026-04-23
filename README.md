# AML-Project

Driver Drowsiness Detection System

An Applied Machine Learning Project

📌 Overview

Driver drowsiness is one of the leading causes of road accidents worldwide. This project presents a real-time driver drowsiness detection system using computer vision and machine learning techniques.

The system monitors the driver's eye movements and alerts them when signs of fatigue or drowsiness are detected.

🎯 Objectives
Detect driver drowsiness in real-time
Reduce accidents caused by fatigue
Build a practical application of machine learning and computer vision
Provide an alert mechanism for safety

🧠 Technologies Used
- Python
- OpenCV (Computer Vision)
- NumPy
- Dlib / Haar Cascades (Face & Eye Detection)
- TensorFlow / Keras (if deep learning model used)
- Scikit-learn (optional for ML models)

⚙️ How It Works
Capture live video using webcam
Detect face and eyes from each frame
Calculate Eye Aspect Ratio (EAR)
Monitor eye closure over time
Trigger alarm if eyes remain closed for a threshold duration

📊 Algorithm
Face Detection
Eye Detection
EAR Calculation
Threshold-based classification
📐 EAR Formula

Eye Aspect Ratio is calculated as:

EAR=
2⋅∣∣p1−p4∣∣
(∣∣p2−p6∣∣+∣∣p3−p5∣∣)
	​

If EAR < threshold → Eyes Closed
If EAR ≥ threshold → Eyes Open

📁 Project Structure
Driver-Drowsiness-Detection/
│
├── dataset/                # Training dataset (if applicable)
├── models/                 # Saved ML/DL models
├── src/                    # Source code
│   ├── detect_drowsiness.py
│   ├── utils.py
│
├── haarcascades/           # Haar cascade XML files
├── alarm.wav               # Alert sound
├── requirements.txt
└── README.md

