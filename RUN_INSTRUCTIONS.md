# ISL AI System - Streamlit Web Application

## Overview

This is a demo-ready Streamlit web interface for the Indian Sign Language Recognition System. The application provides two pipelines:

- **Pipeline 1 (Vision)**: Real-time hand gesture recognition using OpenCV + MediaPipe
- **Pipeline 2 (Speech)**: Speech recognition with Indian language support

## Prerequisites

### 1. Install Streamlit

```bash
pip install -r requirements_app.txt
```

Or directly:
```bash
pip install streamlit>=1.28.0
```

### 2. Verify Virtual Environments

Ensure these virtual environments exist:
- `venv_live/` - For Pipeline 1 (vision-based recognition)
- `venv_pipeline2/` - For Pipeline 2 (speech-based recognition)
- `venv_nlp/` - For sentence normalization (file watcher)
- `venv_tts/` - For Hindi TTS

## Running the Application

### Option 1: Direct Streamlit Run

```bash
cd /Users/sarachhabra/Documents/Sara/ISL_AI_Project
streamlit run app.py
```

### Option 2: With Custom Port

```bash
streamlit run app.py --server.port 8501
```

### Option 3: With Browser Auto-Open Disabled

```bash
streamlit run app.py --server.headless true
```

Then open `http://localhost:8501` manually.

## Pre-Launch Setup

### 1. Start the Sentence Normalizer (Required for Pipeline 1)

The sentence normalizer is a file-watcher that converts ISL keywords to English sentences.

```bash
# In a separate terminal
cd /Users/sarachhabra/Documents/Sara/ISL_AI_Project
source venv_nlp/bin/activate
python run_sentence_normalizer.py
```

**Note:** This requires `ollama` to be running with the `mistral` model.

### 2. Start the Hindi TTS File Watcher (Optional)

If you want the Hindi TTS file watcher to run:

```bash
# In another separate terminal
source venv_tts/bin/activate
python -c "import time; from english_to_hindi_tts_once import *; print('TTS watcher ready')"
```

Or simply run the TTS script when needed - Pipeline 1 triggers it via subprocess.

## Using Pipeline 1 (Vision-based)

1. Click **"Launch Pipeline 1"** on the home page
2. Click **"Start Camera"** to launch the OpenCV webcam window
3. Use keyboard controls in the OpenCV window:

| Key | Action |
|-----|--------|
| `s` | Switch to static mode (SVM) |
| `d` | Switch to dynamic mode (LSTM) |
| `Enter` | Add detected word to sentence |
| `Backspace` | Discard last word |
| `g` | Generate English sentence |
| `h` | Hindi semantic mapping + audio (hidden) |
| `r` | Trigger Hindi TTS |
| `q` | Quit camera |

4. Watch the Streamlit UI for live output updates:
   - ISL Words buffer
   - Generated English sentence
   - Final Hindi/translated output

5. Use Streamlit buttons for:
   - **Stop Camera** - Close the OpenCV window
   - **Reset Sentence** - Clear all text files

## Using Pipeline 2 (Speech-based)

1. Click **"Launch Pipeline 2"** on the home page
2. Select a language:
   - English (US)
   - Hindi
   - Tamil
   - Telugu
   - Kannada
   - Malayalam
   - Bengali
   - Gujarati

3. Click **"Start Listening"** to begin speech recognition

4. Speak in the selected language

5. Click **"Stop & Process"** when finished

6. View results:
   - Original detected text
   - English translation
   - Click **"Generate ISL Video"** to create sign language video

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit App (app.py)                  │
│                  Controller + Viewer Only                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   Pipeline 1        │       │   Pipeline 2        │
│   (Vision-based)    │       │   (Speech-based)    │
│                     │       │                     │
│ Subprocess:         │       │ Subprocess:         │
│ venv_live/bin/python│       │ venv_pipeline2/bin/ │
│ live_fusion_webcam.py│      │ python              │
│                     │       │ speech_to_text.py   │
│ Native cv2 window   │       │                     │
│ with keyboard       │       │                     │
│ controls            │       │                     │
└─────────┬───────────┘       └─────────┬───────────┘
          │                             │
          └─────────────┬───────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │   File Watcher Architecture │
          │                             │
          │   isl_words.txt            │
          │   english_sentence.txt     │
          │   final_sentence.txt       │
          │   hindi_tts.txt            │
          └─────────┬───────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────────┐
│ venv_nlp/       │ │ venv_tts/           │
│ sentence_       │ │ english_to_hindi_   │
│ normalisation.py│ │ tts_once.py         │
│ (ollama/mistral)│ │ (gTTS + afplay)     │
└─────────────────┘ └─────────────────────┘
```

## Troubleshooting

### Camera Won't Start
- Check that no other application is using the webcam
- Ensure `venv_live` has OpenCV and MediaPipe installed

### Speech Recognition Not Working
- Verify microphone permissions in System Preferences > Security & Privacy
- Check that `venv_pipeline2` has `speechrecognition` installed

### Hindi TTS Not Playing
- Ensure `gTTS` and `afplay` (macOS) or `mpg123` (Linux) are available
- Check `venv_tts` has required packages

### Sentence Normalizer Not Working
- Verify `ollama` is running: `ollama serve`
- Check mistral model is installed: `ollama pull mistral`

### Process Won't Stop
- Use Activity Monitor (macOS) or `ps aux | grep python` to find orphaned processes
- Kill manually: `kill -9 <PID>`

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `live_fusion_webcam.py` | Pipeline 1 vision recognition |
| `speech_to_text.py` | Pipeline 2 speech recognition |
| `english_to_isl.py` | Generate ISL video from English |
| `english_to_hindi_tts_once.py` | Hindi TTS generation |
| `run_sentence_normalizer.py` | File-watcher for ISL→English |
| `isl_words.txt` | ISL keywords buffer |
| `english_sentence.txt` | Generated English sentence |
| `final_sentence.txt` | Final translated output |

## Clean Up

To reset the system:

```bash
# Clear all output files
> isl_words.txt
> english_sentence.txt
> final_sentence.txt
> output_isl_sentence.mp4
```

Or click **"Reset Sentence"** in the Streamlit UI.

