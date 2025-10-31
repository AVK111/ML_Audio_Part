import streamlit as st
import os
import sounddevice as sd
import wavio
from Audio_Model import audio_model
from Video_Model import video_model
from Text_Model import text_model

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(page_title="🎯 Unified Multimodal Stress Detection", layout="wide")
st.title("🎯 Unified Multimodal Stress Detection System")
st.markdown("### Detect stress levels using Audio, Video, and Text inputs combined!")

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙️ Settings")
duration = st.sidebar.slider("🎙️ Audio Recording Duration (seconds)", 3, 10, 5)

# ---------------- AUDIO RECORD FUNCTION ----------------
def record_audio(duration=5, filename="temp_audio.wav"):
    fs = 44100  # Sample rate
    st.info(f"🎙️ Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    st.success("✅ Recording saved successfully!")
    return filename

# ---------------- LAYOUT: 3 COLUMNS ----------------
col1, col2, col3 = st.columns(3)

# 🎧 ----------- AUDIO SECTION -----------
with col1:
    st.subheader("🎧 Audio-Based Detection")
    if st.button("🎙️ Record Live Audio"):
        try:
            audio_file = record_audio(duration)
            with st.spinner("Analyzing audio..."):
                audio_score, emotion, stress, depression = audio_model.predict_audio_stress(audio_file)
                st.success("✅ Audio analysis complete!")
                st.write(f"**Emotion:** {emotion}")
                st.write(f"**Stress Level:** {stress}")
                st.write(f"**Depression Indicator:** {depression}")
        except Exception as e:
            st.error(f"⚠️ Audio Model Error: {str(e)}")

# 🎥 ----------- VIDEO SECTION -----------
with col2:
    st.subheader("🎥 Video-Based Detection")
    if st.button("📹 Start Webcam Analysis"):
        try:
            with st.spinner("Starting webcam..."):
                score, stress = video_model.run_video_stress_detection()
                st.success("✅ Video analysis complete!")
                st.write(f"**Video Stress Score:** {score:.2f}")
                st.write(f"**Detected Stress:** {stress}")
        except Exception as e:
            st.error(f"⚠️ Video Model Error: {str(e)}")

# 💬 ----------- TEXT SECTION -----------
with col3:
    st.subheader("💬 Text-Based Detection")
    user_input = st.text_input("How are you feeling right now?")
    if st.button("🧠 Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter how you feel before analysis.")
        else:
            try:
                with st.spinner("Analyzing text..."):
                    emotion, stress, depression, _ = text_model.predict_text_stress(user_input)
                    st.success("✅ Text analysis complete!")
                    st.write(f"**Emotion:** {emotion}")
                    st.write(f"**Stress Level:** {stress}")
                    st.write(f"**Depression Indicator:** {depression}")
            except Exception as e:
                st.error(f"⚠️ Text Model Error: {str(e)}")

# ---------------- FINAL COMBINED SCORE ----------------
st.markdown("---")
st.subheader("🧾 Final Combined Stress Assessment")

if st.button("🧮 Compute Combined Stress Level"):
    st.info("⚙️ Combined scoring will be added soon — for now, test each input separately.")
