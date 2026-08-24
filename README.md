# GestureLink--ISL---Regional-Two-Way-Conversion-


Demo: https://drive.google.com/drive/u/0/folders/1fvkCI3RpWTsxowX3qRkHkokkA7hcXm5M


GestureLink is an AI-powered system designed to enable two-way communication between Indian Sign Language (ISL) users and non-signers.

The system supports two communication pipelines:

- **ISL gestures → spoken language**
- **Regional speech → ISL gestures**

It integrates computer vision, machine learning, speech recognition, and translation into a single interactive interface built with Streamlit.

---

## System Architecture

The project implements two communication pipelines.

### Pipeline 1 — Vision-Based ISL Recognition

```text
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
```

Users perform gestures in front of the webcam. The system detects sign words, forms sentences, and produces speech output.

### Pipeline 2 — Speech to ISL Conversion

```text
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
```

The system listens to spoken input in multiple Indian languages and generates an ISL video representing the sentence.

**Supported languages:**

- English
- Hindi
- Tamil
- Telugu
- Kannada
- Malayalam
- Bengali
- Gujarati

---

## Features

- Real-time ISL gesture recognition using webcam
- Speech-to-text recognition for multiple Indian languages
- Automatic language detection
- Translation to English
- ISL sentence generation
- ISL video output
- Text-to-speech output
- Interactive Streamlit web interface

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Scikit-learn
- SpeechRecognition
- Google Translate API

---

## Project Structure

```text
GestureLink--ISL---Regional-Two-Way-Conversion-
│
├── app.py                        # Streamlit interface
├── launch_demo.py                # Demo launcher
│
├── train_lstm.py                 # LSTM training
├── train_svm.py                  # SVM training
├── train_static_svm.py           # Static gesture classifier
│
├── live_lstm_webcam.py           # Real-time LSTM inference
├── live_fusion_webcam.py         # Gesture fusion pipeline
│
├── speech_to_text.py             # Speech recognition pipeline
├── indic_translate.py             # Language translation
│
├── english_to_isl.py             # Text to ISL conversion
├── english_to_isl_st.py          # ISL sentence generation
│
├── sentence_normalisation.py     # Sentence normalization
├── sentence_map.py               # Word mapping utilities
│
├── requirements_app.txt          # Project dependencies
└── RUN_INSTRUCTIONS.md           # Detailed run instructions
```

---

## How to Run the System

### 1. Clone the Repository

```bash
git clone https://github.com/sara23chhabra/GestureLink--ISL---Regional-Two-Way-Conversion-.git
cd GestureLink--ISL---Regional-Two-Way-Conversion-
```

### 2. Install Dependencies

```bash
pip install -r requirements_app.txt
```


### Prerequisites

The current application was developed and tested using **Python 3.9**.

The project uses separate virtual environments because some components have different dependency requirements.

The environments used by the application are:

| Environment | Purpose |
|-------------|---------|
| `venv_live` | Streamlit, MediaPipe, OpenCV, TensorFlow/Keras and gesture recognition |
| `venv_pipeline2` | Speech recognition pipeline |
| `venv_nlp` | Sentence normalization using Ollama |
| `venv_tts` | Hindi text-to-speech |

The main `app.py` application automatically invokes the appropriate environment for the individual pipelines.

### Start the Application

Activate the environment containing the Streamlit application:

```bash
source venv_live/bin/activate
```

Then launch Streamlit:

```bash
streamlit run app.py
```

The application will open in the browser.

> **Note:** The virtual environments listed above are local development environments and are not included in the GitHub repository. A new installation requires the corresponding dependencies to be configured separately.

---

## Keyboard Controls

When the OpenCV camera window is active:

| Key | Function |
|-----|----------|
| `s` | Static gesture mode |
| `d` | Dynamic gesture mode |
| `Enter` | Add detected word |
| `Backspace` | Discard word |
| `g` | Generate sentence |
| `r` | Hindi text-to-speech |
| `q` | Quit camera |

---

## Training Models

The repository includes training scripts for gesture classification:

- `train_lstm.py` — LSTM training
- `train_svm.py` — SVM training
- `train_static_svm.py` — Static gesture classification

These scripts can be used to retrain the models with new datasets.

> **Note:** Pretrained models are not included due to GitHub size limits. They can be regenerated using the provided training scripts.

---

## Applications

- Assistive communication for the deaf community
- Sign language learning tools
- Accessibility technologies
- Human-computer interaction systems

---

## Author

**Sara Chhabra**
