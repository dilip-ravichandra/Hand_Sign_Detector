# Hand_Sign_Detector
# 🤟 Real-Time Custom Sign Language Detector

A real-time sign language detection system built using **TensorFlow**, **MediaPipe**, and **Scikit-learn**. This project captures and classifies custom hand gestures using hand landmarks for real-time applications.

## 🚀 Features

- Live sign language detection using webcam
- Custom sign training (Hello, Yes, No, Thank You, I Love You)
- Landmark-based detection using MediaPipe Hands
- Lightweight and real-time prediction
- Easily extendable for new gestures

## 🧠 Technologies Used

- Python 3.10
- TensorFlow
- MediaPipe
- OpenCV
- Scikit-learn
- NumPy
- Pickle

## 📁 Project Structure

├── collect_image.py # Data collection via webcam
├── train_model.py # Model training & saving
├── detect.py # Real-time detection script
├── MP_Data/ # Collected gesture landmarks
├── sign_language_model.h5 # Trained model
├── label_encoder.pkl # Label encoder
└── utils/
└── hand_tracking_module.py

bash
Copy
Edit

## 🎥 Demo

![Demo](assets/demo.gif)

## 📦 Installation

```bash
git clone https://github.com/Dilip-Ravichandra/SignLanguageDetector.git
cd SignLanguageDetector
pip install -r requirements.txt
📸 Collecting Custom Signs
bash
Copy
Edit
python collect_image.py
Press h multiple times to capture each frame

Press q to switch to the next gesture

🧠 Training the Model
bash
Copy
Edit
python train_model.py
🎯 Running Real-Time Detection
bash
Copy
Edit
python detect.py
🙋‍♂️ Author
Dilip R
📍 RV University
🔗 LinkedIn
📧 Email: [your.email@example.com]

⭐ Contribute
Pull requests are welcome. For major changes, please open an issue first.

yaml
Copy
Edit

---

### ✅ **Step 3: `requirements.txt`**

```txt
mediapipe==0.10.0
opencv-python
numpy
scikit-learn
tensorflow==2.8.0

Added script to collect custom sign landmarks

Trained Random Forest model for custom signs

Implemented real-time detection using webcam

Updated README and demo assets

✅ Step 5: Push to GitHub
bash
Copy
Edit
git init
git add .
git commit -m "Initial commit - Sign Language Detector"
git remote add origin https://github.com/yourusername/SignLanguageDetector.git
git push -u origin main
✅ Step 6: Add GitHub Topics & Tags
sign-language

tensorflow

mediapipe

computer-vision

gesture-recognition

AI-for-good
