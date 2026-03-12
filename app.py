#!/usr/bin/env python3
"""
ISL AI System - Streamlit Web Application
==========================================
A demo-ready UI for Indian Sign Language Recognition System.

PIPELINE 1 (Vision-based):
- Opens native OpenCV window for webcam
- Keyboard controls work via cv2.waitKey()
- Streamlit displays output file contents

PIPELINE 2 (Speech-based):
- Launches speech recognition subprocess
- Displays detected language, original text, and translation

Author: ISL AI Project
"""

import streamlit as st
import subprocess
import os
import time
import signal
import sys
from pathlib import Path
import speech_recognition as sr
from googletrans import Translator



# ======================================================
# CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="ISL Recognition System",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Virtual environment paths
VENV_LIVE = PROJECT_ROOT / "venv_live" / "bin" / "python"
VENV_PIPELINE2 = PROJECT_ROOT / "venv_pipeline2" / "bin" / "python"
VENV_TTS = PROJECT_ROOT / "venv_tts" / "bin" / "python"

# File paths for IPC
ISL_WORDS_FILE = PROJECT_ROOT / "isl_words.txt"
ENGLISH_SENTENCE_FILE = PROJECT_ROOT / "english_sentence.txt"
FINAL_SENTENCE_FILE = PROJECT_ROOT / "final_sentence.txt"

# Pipeline scripts
LIVE_FUSION_SCRIPT = PROJECT_ROOT / "live_manual_sentence_demo.py"
SPEECH_TO_TEXT_SCRIPT = PROJECT_ROOT / "speech_to_text.py"
ENGLISH_TO_ISL_SCRIPT = PROJECT_ROOT / "english_to_isl_st.py"
HINDI_TTS_SCRIPT = PROJECT_ROOT / "english_to_hindi_tts_once.py"

# ======================================================
# SESSION STATE
# ======================================================
if 'pipeline1_process' not in st.session_state:
    st.session_state.pipeline1_process = None
if 'pipeline1_pid' not in st.session_state:
    st.session_state.pipeline1_pid = None
if 'pipeline2_process' not in st.session_state:
    st.session_state.pipeline2_process = None
if 'pipeline2_pid' not in st.session_state:
    st.session_state.pipeline2_pid = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def read_file_content(filepath: Path) -> str:
    """Read and return file content, or empty string if file doesn't exist."""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        st.error(f"Error reading {filepath.name}: {e}")
    return ""


def clear_text_files():
    """Clear all text files used for IPC."""
    files_to_clear = [ISL_WORDS_FILE, ENGLISH_SENTENCE_FILE, FINAL_SENTENCE_FILE]
    for f in files_to_clear:
        if f.exists():
            with open(f, 'w') as file:
                file.write("")
            st.toast(f"Cleared {f.name}", icon="🗑️")


def terminate_process(pid: int):
    """Terminate a process by PID, handling both Unix and macOS."""
    try:
        # Try SIGTERM first for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.2)
        # Check if process is still running
        try:
            os.kill(pid, 0)  # This will raise OSError if process doesn't exist
            # Process still running, force kill
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # Process already terminated
        return True
    except Exception as e:
        st.warning(f"Could not terminate process {pid}: {e}")
        return False


def stop_pipeline1():
    """Stop Pipeline 1 (Vision)."""
    if st.session_state.pipeline1_process:
        try:
            pid = st.session_state.pipeline1_pid
            if pid:
                terminate_process(pid)
            st.session_state.pipeline1_process.terminate()
            st.session_state.pipeline1_process.wait(timeout=5)
        except Exception as e:
            st.warning(f"Pipeline 1 termination: {e}")
        st.session_state.pipeline1_process = None
        st.session_state.pipeline1_pid = None
        st.toast("Camera stopped", icon="📷")


def stop_pipeline2():
    """Stop Pipeline 2 (Speech)."""
    if st.session_state.pipeline2_process:
        try:
            pid = st.session_state.pipeline2_pid
            if pid:
                terminate_process(pid)
            st.session_state.pipeline2_process.terminate()
            st.session_state.pipeline2_process.wait(timeout=5)
        except Exception as e:
            st.warning(f"Pipeline 2 termination: {e}")
        st.session_state.pipeline2_process = None
        st.session_state.pipeline2_pid = None
        st.toast("Speech listening stopped", icon="🎤")


def start_pipeline1():
    """Start Pipeline 1 - Vision-based ISL recognition."""
    if not LIVE_FUSION_SCRIPT.exists():
        st.error(f"Script not found: {LIVE_FUSION_SCRIPT}")
        return None
    
    if not VENV_LIVE.exists():
        st.error(f"Virtual environment not found: {VENV_LIVE}")
        return None
    
    try:
        process = subprocess.Popen(
            [str(VENV_LIVE), str(LIVE_FUSION_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for clean termination
        )
        st.toast("Camera started! Check the OpenCV window.", icon="📷")
        return process
    except Exception as e:
        st.error(f"Failed to start Pipeline 1: {e}")
        return None


def start_pipeline2_mode(language: str, mode: str = "interactive"):
    """Start Pipeline 2 - Speech-based ISL recognition."""
    if not SPEECH_TO_TEXT_SCRIPT.exists():
        st.error(f"Script not found: {SPEECH_TO_TEXT_SCRIPT}")
        return None, None
    
    if not VENV_PIPELINE2.exists():
        st.error(f"Virtual environment not found: {VENV_PIPELINE2}")
        return None, None
    
    try:
        # Set environment variables for language
        env = os.environ.copy()
        env['ISL_LANGUAGE'] = language
        env['ISL_MODE'] = mode
        
        process = subprocess.Popen(
            [str(VENV_PIPELINE2), str(SPEECH_TO_TEXT_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid
        )
        return process, process.pid
    except Exception as e:
        st.error(f"Failed to start Pipeline 2: {e}")
        return None, None


def trigger_hindi_tts():
    """Trigger Hindi TTS via subprocess."""
    if not VENV_TTS.exists():
        st.error(f"Virtual environment not found: {VENV_TTS}")
        return
    
    try:
        subprocess.Popen(
            [str(VENV_TTS), str(HINDI_TTS_SCRIPT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        st.toast("Hindi TTS triggered!", icon="🔊")
    except Exception as e:
        st.error(f"Failed to trigger TTS: {e}")


def generate_isl_video(english_text: str, original_text: str = ""):
    video_path = PROJECT_ROOT / "output_isl_sentence.mp4"

    # Remove stale cached video
    if video_path.exists():
        video_path.unlink()

    with open(ENGLISH_SENTENCE_FILE, "w", encoding="utf-8") as f:
        f.write(english_text)

    subprocess.run(
        [str(VENV_PIPELINE2), str(ENGLISH_TO_ISL_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        check=True
    )

    time.sleep(1.0)  # allow OpenCV to finalize MP4




# ======================================================
# PAGE: HOME / LANDING
# ======================================================
def show_home_page():
    st.markdown("""
    <div style="text-align:center; margin-bottom:40px;">
        <h1 style="font-weight:700;">Recognition System</h1>
        <p style="color:#aaa;">Choose a pipeline to begin</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    card_style = """
    background: linear-gradient(135deg, #5f5cff 0%, #7fd8be 100%);
    height: 220px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: 600;
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
    """

    button_style = """
    margin-top: 16px;
    """

    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            🤟 ISL to<br>Regional Speech
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='{button_style}'>", unsafe_allow_html=True)
        if st.button("Launch", key="launch_p1", use_container_width=True):
            st.session_state.current_page = "pipeline1"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            🎙 Regional Speech<br>to ISL
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='{button_style}'>", unsafe_allow_html=True)
        if st.button("Launch", key="launch_p2", use_container_width=True):
            st.session_state.current_page = "pipeline2"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:50px; color:#666; font-size:12px;">
        ISL AI Project · Academic Demo
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE: PIPELINE 1 (VISION)
# ======================================================
def show_pipeline1_page():
    """Display Pipeline 1 - Vision-based ISL recognition."""
    
    # Header with back button
    col_header1, col_header2 = st.columns([1, 6])
    with col_header1:
        if st.button("← Back", use_container_width=True):
            stop_pipeline1()
            st.session_state.current_page = 'home'
            st.rerun()
    with col_header2:
        st.markdown("### 📷 Pipeline 1: Vision-based ISL Recognition")
    
    # Control buttons
    st.markdown("#### Camera Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Start Camera", type="primary", use_container_width=True):
            if st.session_state.pipeline1_process is None:
                process = start_pipeline1()
                if process:
                    st.session_state.pipeline1_process = process
                    st.session_state.pipeline1_pid = process.pid
                    time.sleep(1)  # Give process time to start
    
    with col2:
        if st.button("Stop Camera", use_container_width=True):
            stop_pipeline1()
    
    with col3:
        if st.button("Generate Sentence(press g on openCV)", use_container_width=True):
            # This simulates pressing 'g' - write to isl_words.txt
            # The live_fusion_webcam.py will detect this
            current_words = read_file_content(ISL_WORDS_FILE)
            st.toast(f"Sentence generation triggered", icon="📝")
    
    with col4:
        if st.button("Reset Sentence", use_container_width=True):
            clear_text_files()
    
    # Hindi TTS button
    st.markdown("---")
    col_tts1, col_tts2 = st.columns([1, 6])
    with col_tts1:
        if st.button("🔊 Hindi TTS (r)", use_container_width=True):
            trigger_hindi_tts()
    
    # Live output display
    st.markdown("---")
    st.markdown("#### Live Output")
    
    # Create a container for live updates
    output_container = st.container()
    
    with output_container:
        # Read current file contents
        isl_words = read_file_content(ISL_WORDS_FILE)
        english_sentence = read_file_content(ENGLISH_SENTENCE_FILE)
        final_sentence = read_file_content(FINAL_SENTENCE_FILE)
        
        # Display in a formatted way
        col_display1, col_display2, col_display3 = st.columns(3)
        
        with col_display1:
            st.markdown("**ISL Words:**")
            st.info(isl_words if isl_words else "Waiting for input...")
        
        with col_display2:
            st.markdown("**English Sentence:**")
            st.success(english_sentence if english_sentence else "Waiting for generation...")
        
        with col_display3:
            st.markdown("**Output:**")
            st.warning(final_sentence if final_sentence else "Waiting...")
    
    # Keyboard controls reference
    with st.expander("ℹ️ Keyboard Controls (Hidden - for demo operator)"):
        st.markdown("""
        | Key | Action |
        |-----|--------|
        | `s` | Static mode |
        | `d` | Dynamic mode |
        | `Enter` | Add detected word |
        | `Backspace` | Discard word |
        | `g` | Generate sentence |
        | `h` | Hindi semantic mapping (no UI) |
        | `r` | Hindi TTS |
        | `q` | Quit camera |
        
        **Note:** These controls work in the native OpenCV window, not in Streamlit.
        """)
    
    # Auto-refresh for output display
    time.sleep(0.5)
    st.rerun()


# ======================================================
# PAGE: PIPELINE 2 (SPEECH)
# ======================================================
def show_pipeline2_page():
    """Display Pipeline 2 - Speech-based ISL recognition."""
    
    # Header with back button
    col_header1, col_header2 = st.columns([1, 6])
    with col_header1:
        if st.button("← Back", use_container_width=True):
            stop_pipeline2()
            st.session_state.current_page = 'home'
            st.rerun()
    with col_header2:
        st.markdown("### 🎤 Pipeline 2: Speech-based ISL Recognition")
    
    # Language selection
    st.markdown("#### Select Language")
    languages = {
        "en-US": "English (US)",
        "hi-IN": "Hindi",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "bn-IN": "Bengali",
        "gu-IN": "Gujarati"
    }
    
    if 'pipeline2_language' not in st.session_state:
        st.session_state.pipeline2_language = "hi-IN"
    
    lang_cols = st.columns(4)
    for i, (code, name) in enumerate(languages.items()):
        with lang_cols[i % 4]:
            if st.button(name, 
                        type="primary" if st.session_state.pipeline2_language == code else "secondary",
                        use_container_width=True):
                st.session_state.pipeline2_language = code
    
    # Listening controls
    st.markdown("---")
    st.markdown("#### Speech Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Listening", type="primary", use_container_width=True):

            recognizer = sr.Recognizer()
            microphone = sr.Microphone()
            translator = Translator()

            with microphone as source:
                st.info("Listening... Speak now")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                audio = recognizer.listen(source, phrase_time_limit=8)

            try:
                detected_text = recognizer.recognize_google(
                    audio,
                    language=st.session_state.pipeline2_language
                )

                if st.session_state.pipeline2_language.startswith("en"):
                    english_text = detected_text
                else:
                    english_text = translator.translate(
                        detected_text,
                        dest="en"
                    ).text

                # Write outputs for downstream pipeline
                with open(ISL_WORDS_FILE, "w", encoding="utf-8") as f:
                    f.write(detected_text)

                with open(ENGLISH_SENTENCE_FILE, "w", encoding="utf-8") as f:
                    f.write(english_text)

                st.success("Speech processed successfully")

            except Exception as e:
                st.error(f"Speech recognition failed: {e}")

        
    with col2:
        if st.button("⏹ Stop & Process", use_container_width=True):
            stop_pipeline2()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            clear_text_files()
            st.toast("Text files cleared", icon="🗑️")
    
    # Output display
    st.markdown("---")
    st.markdown("#### Speech Output")
    
    # Read current outputs
    original_text = read_file_content(ISL_WORDS_FILE)  # Speech script writes to isl_words.txt
    english_text = read_file_content(ENGLISH_SENTENCE_FILE)
    
    # Display results
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown(f"**Detected ({languages.get(st.session_state.pipeline2_language, 'Unknown')}):**")
        st.info(original_text if original_text else "Speak to see detected text...")
    
    with col_out2:
        st.markdown("**English Translation:**")
        st.success(english_text if english_text else "Translation will appear here...")
    
    # ISL Video Generation
    if english_text:
        st.markdown("---")
        st.markdown("#### ISL Video")
        if st.button("🎬 Generate ISL Video"):
            with st.spinner("Generating ISL video..."):
                generate_isl_video(english_text, original_text)
                time.sleep(0.5)  # allow file flush
            st.success("ISL video generated")

        
        # Check if video exists and display
        video_path = PROJECT_ROOT / "output_isl_sentence.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                st.video(f.read())
        else :
            st.info("ISL video will appear here after generation...")

    

    


# ======================================================
# MAIN APP
# ======================================================
def main():
    """Main application entry point."""
    
    # Custom CSS for cleaner UI
    st.markdown("""
    <style>
    .stButton button {
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Route to current page
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'pipeline1':
        show_pipeline1_page()
    elif st.session_state.current_page == 'pipeline2':
        show_pipeline2_page()


if __name__ == "__main__":
    main()

