import cv2
import numpy as np
import dlib
import imutils
from scipy.spatial import distance as dist
from collections import deque
import os
import urllib.request
import bz2

class EnhancedStressDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.initialize_predictor()

        # Parameters
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 3

        # Calibration
        self.baseline_ear = 0.3
        self.baseline_mar = 0.2
        self.calibration_frames = 0
        self.is_calibrated = False
        self.calibration_duration = 60

        # Histories
        self.ear_history = deque(maxlen=100)
        self.blink_count = 0
        self.stress_levels = deque(maxlen=200)

        self.blink_frames = 0

    def initialize_predictor(self):
        """Load or download dlib face landmark model"""
        predictor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shape_predictor_68_face_landmarks.dat"))
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        try:
            if not os.path.exists(predictor_path):
                print("Downloading facial landmark predictor...")
                urllib.request.urlretrieve(url, predictor_path + ".bz2")
                with bz2.BZ2File(predictor_path + ".bz2", 'rb') as fr:
                    with open(predictor_path, 'wb') as fw:
                        fw.write(fr.read())
                os.remove(predictor_path + ".bz2")
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Facial landmark predictor loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading predictor: {e}")

    def get_eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)

    def calibrate_baseline(self, ear, mar):
        if self.calibration_frames < self.calibration_duration:
            self.baseline_ear = (self.baseline_ear * self.calibration_frames + ear) / (self.calibration_frames + 1)
            self.baseline_mar = (self.baseline_mar * self.calibration_frames + mar) / (self.calibration_frames + 1)
            self.calibration_frames += 1
        else:
            self.is_calibrated = True

    def detect_blinks(self, ear):
        self.ear_history.append(ear)
        if len(self.ear_history) >= self.CONSECUTIVE_FRAMES:
            recent_ears = list(self.ear_history)[-self.CONSECUTIVE_FRAMES:]
            if all(e < self.EAR_THRESHOLD for e in recent_ears):
                if self.blink_frames == 0:
                    self.blink_count += 1
                    self.blink_frames = self.CONSECUTIVE_FRAMES
            else:
                self.blink_frames = max(0, self.blink_frames - 1)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        stress_level = 0
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]
            ear = (self.get_eye_aspect_ratio(left_eye) + self.get_eye_aspect_ratio(right_eye)) / 2.0
            mar = self.get_mouth_aspect_ratio(mouth)
            self.detect_blinks(ear)

            if not self.is_calibrated:
                self.calibrate_baseline(ear, mar)
            else:
                ear_dev = abs(ear - self.baseline_ear)
                mar_dev = abs(mar - self.baseline_mar)
                stress_level = (ear_dev + mar_dev) * 100
                self.stress_levels.append(stress_level)
        return frame, stress_level


def run_video_stress_detection():
    """Wrapper function used by main_combined_model.py"""
    print("🎥 Starting video stress detection...")
    detector = EnhancedStressDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return 0.0

    total_stress = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frame, stress = detector.process_frame(frame)
        total_stress.append(stress)

        # ⚠️ Skip GUI window to avoid OpenCV error
        # cv2.imshow("Video Stress Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()  # Not needed in headless mode

    avg_stress = np.mean(total_stress) / 100 if total_stress else 0.3
    print(f"✅ Video Stress Score: {avg_stress:.2f}")
    return avg_stress, "High" if avg_stress > 0.5 else "Low"


if __name__ == "__main__":
    run_video_stress_detection()
