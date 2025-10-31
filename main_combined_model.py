import sys
import os
import time

import sys
from Audio_Model import audio_model as Audio_Model
from Video_Model import video_model as Video_Model
from Text_Model import text_model as Text_Model

print("\n🎯 Unified Multimodal Stress Detection System")
print("============================================\n")

# 🎙️ --- AUDIO MODEL ---
try:
    audio_path = input("🎙️ Enter path to your audio (.wav) file: ").strip()
    print("\n🎙️ Running Audio Model...")
    audio_emotion, audio_stress, audio_depression, audio_score = Audio_Model.predict_audio_stress(audio_path)
    print(f"🎧 Predicted Emotion (Audio): {audio_emotion}")
    print(f"🧠 Stress Level: {audio_stress}")
    print(f"💭 Depression Indicator: {audio_depression}")
    print("✅ Audio Model Finished Successfully!\n")
except Exception as e:
    print(f"⚠️ Audio Model Error: {e}\n")
    audio_score = 0

# 🎥 --- VIDEO MODEL ---
try:
    print("🎥 Running Video Model...")
    video_score, video_stress = Video_Model.run_video_stress_detection()
    print(f"✅ Video Stress Score: {video_score}")
    print(f"🧠 Stress Level (Video): {video_stress}")
    print("✅ Video Model Finished Successfully!\n")
except Exception as e:
    print(f"⚠️ Video Model Error: {e}\n")
    video_score = 0

# 💬 --- TEXT MODEL ---
try:
    user_text = input("💬 Enter how you feel (e.g., 'I am tired and anxious'): ").strip()
    print("\n💬 Running Text Model...")
    text_emotion, text_stress, text_depression, text_score = Text_Model.predict_text_stress(user_text)
    print(f"💬 Input Text: {user_text}")
    print(f"🪞 Predicted Emotion: {text_emotion}")
    print(f"🧠 Stress Level: {text_stress}")
    print(f"💭 Depression Indicator: {text_depression}")
    print("✅ Text Model Finished Successfully!\n")
except Exception as e:
    print(f"⚠️ Text Model Error: {e}\n")
    text_score = 0

# 🧾 --- FINAL COMBINED RESULT ---
final_stress_level = (audio_score + video_score + text_score) / 3
print(f"🧾 Final Combined Stress Level: {final_stress_level:.2f}")

if final_stress_level < 0.4:
    print("🙂 Low Stress — You seem calm.")
elif final_stress_level < 0.7:
    print("😐 Moderate Stress — Try taking a break.")
else:
    print("😟 High Stress — Consider talking to someone or relaxing.\n")
