# audio_utils.py
import sounddevice as sd
import wavio

def record_audio(duration=5, filename="temp_audio.wav"):
    print("🎙️ Recording... Speak now!")
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"✅ Recording saved as {filename}")
    return filename
