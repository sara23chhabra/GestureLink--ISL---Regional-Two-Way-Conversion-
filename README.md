# GestureLink--ISL---Regional-Two-Way-Conversion-


Demo: https://drive.google.com/drive/u/0/folders/1fvkCI3RpWTsxowX3qRkHkokkA7hcXm5M

GestureLink: ISL ↔ Regional Language Two-Way Communication System
GestureLink is an AI-powered system designed to enable two-way communication between Indian Sign Language (ISL) users and non-signers.
The system supports both:
ISL gestures → spoken language
Regional speech → ISL gestures
It integrates computer vision, machine learning, speech recognition, and translation into a single interactive interface built with Streamlit.

System Architecture
The project implements two communication pipelines.


Pipeline 1 — Vision-Based ISL Recognition
Webcam Input
      ↓
Hand Landmark Detection (MediaPipe)
      ↓
Feature Extraction
      ↓
Gesture Classification (LSTM / SVM)
      ↓
Word Sequence Generation
      ↓
Sentence Normalization
      ↓
Text Output
      ↓
Hindi Text-to-Speech

Users perform gestures in front of the webcam and the system detects sign words, forms sentences, and produces speech output.


Pipeline 2 — Speech to ISL Conversion
Speech Input
      ↓
Speech Recognition
      ↓
Language Detection
      ↓
Translation to English
      ↓
Sentence Processing
      ↓
ISL Video Generation

The system listens to spoken input in multiple Indian languages and generates an ISL video representing the sentence. 
Supported languages include: English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati


Features
Real-time ISL gesture recognition using webcam
Speech-to-text recognition for multiple Indian languages
Translation to English
ISL sentence generation
ISL video output
Text-to-speech output
Interactive Streamlit web interface

Technologies Used
Python
Streamlit
OpenCV
MediaPipe
TensorFlow / Keras
Scikit-learn
SpeechRecognition
Google Translate API


Project Structure 
GestureLink--ISL---Regional-Two-Way-Conversion-

app.py                        # Streamlit interface
launch_demo.py                # Demo launcher

train_lstm.py                 # LSTM training
train_svm.py                  # SVM training
train_static_svm.py           # Static gesture classifier

live_lstm_webcam.py           # Real-time LSTM inference
live_fusion_webcam.py         # Gesture fusion pipeline

speech_to_text.py             # Speech recognition pipeline
indic_translate.py            # Language translation

english_to_isl.py             # Text to ISL conversion
english_to_isl_st.py          # ISL sentence generation

sentence_normalisation.py     # Sentence normalization
sentence_map.py               # Word mapping utilities

requirements_app.txt          # Project dependencies
RUN_INSTRUCTIONS.md           # Detailed run instructions



How to Run the System
1. Clone the repository
   git clone https://github.com/sara23chhabra/GestureLink--ISL---Regional-Two-Way-Conversion-.git
   cd GestureLink--ISL---Regional-Two-Way-Conversion-

2. Install dependencies
   pip install -r requirements_app.txt

3. Launch the application
   streamlit run app.py


Keyboard Controls (Vision Pipeline)
When the OpenCV camera window is active:
Key	            Function
s	          Static gesture mode
d	          Dynamic gesture mode
Enter	      Add detected word
Backspace	  Discard word
g	          Generate sentence
r	          Hindi text-to-speech
q	          Quit camera

Training Models
The repository includes training scripts for gesture classification:
train_lstm.py
train_svm.py
train_static_svm.py
These scripts can be used to retrain models with new datasets.
Pretrained models are not included due to GitHub size limits and can be regenerated using the provided training scripts.

Applications
Assistive communication for the deaf community
Sign language learning tools
Accessibility technologies
Human-computer interaction systems

Author
Sara Chhabra
