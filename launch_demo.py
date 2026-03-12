#!/usr/bin/env python3
"""
ISL AI System - Demo Launcher
=============================
Launches all required processes for a complete demo:
1. Sentence normalizer (file watcher)
2. Streamlit web application

Usage:
    python launch_demo.py           # Launch with default settings
    python launch_demo.py --skip-nlp # Skip NLP file watcher
    python launch_demo.py --port 8501 # Custom port
"""

import argparse
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Virtual environment paths
VENV_NLP = PROJECT_ROOT / "venv_nlp" / "bin" / "python"
VENV_TTS = PROJECT_ROOT / "venv_tts" / "bin" / "python"

# Scripts
NLP_SCRIPT = PROJECT_ROOT / "run_sentence_normalizer.py"
TTS_SCRIPT = PROJECT_ROOT / "english_to_hindi_tts_once.py"
APP_SCRIPT = PROJECT_ROOT / "app.py"

# Process handles
nlp_process = None
streamlit_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Shutting down all processes...")
    
    if nlp_process:
        print("  - Stopping NLP normalizer...")
        nlp_process.terminate()
        nlp_process.wait(timeout=5)
    
    if streamlit_process:
        print("  - Stopping Streamlit...")
        streamlit_process.terminate()
        streamlit_process.wait(timeout=5)
    
    print("\n✅ All processes stopped. Goodbye!")
    sys.exit(0)


def print_banner():
    """Print the demo banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        👋 Indian Sign Language Recognition System            ║
║                    Demo Launcher v1.0                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_requirements():
    """Check if required files and environments exist."""
    print("📋 Checking requirements...")
    
    all_good = True
    
    # Check virtual environments
    if not VENV_NLP.exists():
        print(f"  ⚠️  NLP venv not found: {VENV_NLP}")
        all_good = False
    else:
        print(f"  ✅ NLP venv found")
    
    # Check scripts
    if not NLP_SCRIPT.exists():
        print(f"  ⚠️  NLP script not found: {NLP_SCRIPT}")
        all_good = False
    else:
        print(f"  ✅ NLP script found")
    
    if not APP_SCRIPT.exists():
        print(f"  ⚠️  App script not found: {APP_SCRIPT}")
        all_good = False
    else:
        print(f"  ✅ App script found")
    
    # Check streamlit
    try:
        result = subprocess.run(["streamlit", "--version"], capture_output=True, text=True)
        print(f"  ✅ Streamlit: {result.stdout.strip()}")
    except FileNotFoundError:
        print(f"  ❌ Streamlit not found. Install with: pip install streamlit")
        all_good = False
    
    return all_good


def start_nlp_normalizer():
    """Start the sentence normalizer file watcher."""
    global nlp_process
    
    print("\n🚀 Starting NLP normalizer...")
    
    try:
        nlp_process = subprocess.Popen(
            [str(VENV_NLP), str(NLP_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait a moment and check if it started
        time.sleep(1)
        if nlp_process.poll() is not None:
            print("  ❌ NLP normalizer failed to start")
            return False
        
        print("  ✅ NLP normalizer running (PID: {})".format(nlp_process.pid))
        return True
        
    except Exception as e:
        print(f"  ❌ Error starting NLP normalizer: {e}")
        return False


def start_streamlit(port=8501, headless=False):
    """Start the Streamlit web application."""
    global streamlit_process
    
    print("\n🚀 Starting Streamlit application...")
    
    cmd = ["streamlit", "run", str(APP_SCRIPT), "--server.port", str(port)]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    try:
        streamlit_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for streamlit to start
        time.sleep(3)
        if streamlit_process.poll() is not None:
            print("  ❌ Streamlit failed to start")
            return False
        
        print("  ✅ Streamlit running (PID: {})".format(streamlit_process.pid))
        print(f"\n🌐 Open your browser to: http://localhost:{port}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error starting Streamlit: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ISL AI System Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_demo.py                    # Normal launch
  python launch_demo.py --skip-nlp         # Skip NLP file watcher
  python launch_demo.py --port 8080        # Custom port
  python launch_demo.py --headless         # Headless mode
        """
    )
    
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip starting the NLP file watcher"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit (default: 8501)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode (no auto-open browser)"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print banner
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Start NLP normalizer (unless skipped)
    if not args.skip_nlp:
        if not start_nlp_normalizer():
            print("\n⚠️  NLP normalizer failed, but continuing...")
    else:
        print("\n⏭️  Skipping NLP normalizer (--skip-nlp)")
    
    # Start Streamlit
    if not start_streamlit(port=args.port, headless=args.headless):
        print("\n❌ Failed to start Streamlit. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All systems ready!")
    print("=" * 60)
    print("""
📝 Instructions:
   1. The Streamlit app is running at http://localhost:{port}
   2. Choose a pipeline from the home page
   3. Follow the on-screen instructions
   
🛑 Press Ctrl+C to stop all processes
    """.format(port=args.port))
    
    # Wait for processes
    try:
        while True:
            time.sleep(1)
            
            # Check if NLP process died
            if nlp_process and nlp_process.poll() is not None:
                print("\n⚠️  NLP normalizer stopped unexpectedly. Restarting...")
                start_nlp_normalizer()
            
            # Check if Streamlit process died
            if streamlit_process and streamlit_process.poll() is not None:
                print("\n⚠️  Streamlit stopped. Exiting...")
                break
                
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()

