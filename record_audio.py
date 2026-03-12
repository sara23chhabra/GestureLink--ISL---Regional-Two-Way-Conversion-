import sounddevice as sd
import soundfile as sf
import os

FS = 44100        # Sample rate
DURATION = 4      # Seconds per recording

AUDIO_MAP = {
    
    "thief_here": "यहाँ चोर है",
    "accident_doctor": "दुर्घटना हुई है, डॉक्टर को बुलाइए",
    "pain_here": "यहाँ दर्द है",


}

os.makedirs("audio", exist_ok=True)

print("🎙️ ISL AUDIO RECORDING STARTED")
print("Press Ctrl+C anytime to stop\n")

for key, hindi in AUDIO_MAP.items():
    filename = f"audio/{key}.wav"

    print("──────────────────────────────")
    print("File:", filename)
    print("Speak this Hindi sentence:")
    print("👉", hindi)
    input("Press ENTER to start recording...")

    audio = sd.rec(int(FS * DURATION), samplerate=FS, channels=1)
    sd.wait()

    sf.write(filename, audio, FS)
    print("✅ Saved:", filename)

print("\n🎉 ALL AUDIO FILES RECORDED")
