import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from collections import deque
import os
import urllib.request
import bz2
import time


class ImprovedStressDetector:
    """Improved stress detector that distinguishes smiles from stress and detects anger"""
    
    def __init__(self):
        print(" Initializing Improved Stress Detector...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.load_predictor()
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.BLINK_THRESHOLD = 3
        
        # Baseline values
        self.baseline_ear = 0.3
        self.baseline_mar = 0.2
        self.baseline_smile = 0.0
        self.baseline_brow_dist = 20.0
        
        # Calibration
        self.calibration_frames = 0
        self.calibrated = False
        self.calibration_needed = 90  # 3 seconds at 30fps
        
        # History tracking
        self.ear_history = deque(maxlen=100)
        self.mar_history = deque(maxlen=100)
        self.smile_history = deque(maxlen=100)
        self.blink_count = 0
        self.blink_frames = 0
        
        # Stress indicators
        self.stress_history = deque(maxlen=200)
        self.rapid_blink_count = 0
        self.last_blink_time = time.time()
        
        # Accuracy tracking
        self.ground_truth = []
        self.predictions = []
        
    def load_predictor(self):
        """Load facial landmark predictor"""
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        if not os.path.exists(predictor_path):
            print(" Downloading facial landmark model (99.7 MB)...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            try:
                urllib.request.urlretrieve(url, predictor_path + ".bz2")
                print(" Extracting...")
                with bz2.BZ2File(predictor_path + ".bz2", 'rb') as fr:
                    with open(predictor_path, 'wb') as fw:
                        fw.write(fr.read())
                os.remove(predictor_path + ".bz2")
            except Exception as e:
                print(f" Download failed: {e}")
                raise
        
        self.predictor = dlib.shape_predictor(predictor_path)
        print(" Model loaded!")
    
    def get_ear(self, eye):
        """Calculate Eye Aspect Ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def get_mar(self, mouth):
        """Calculate Mouth Aspect Ratio"""
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)
    
    def detect_smile(self, mouth, left_eye, right_eye):
        """Detect GENUINE smile (Duchenne smile) - both mouth AND eyes"""
        # Mouth analysis
        left_corner = mouth[0]
        right_corner = mouth[6]
        top_lip = mouth[3]
        bottom_lip = mouth[9]
        
        mouth_width = dist.euclidean(left_corner, right_corner)
        mouth_height = dist.euclidean(top_lip, bottom_lip)
        mouth_ratio = mouth_width / (mouth_height + 1e-6)
        
        # Check if corners are raised (genuine smile)
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        center_y = (top_lip[1] + bottom_lip[1]) / 2
        corners_raised = corner_avg_y > center_y
        
        # Eye analysis - genuine smiles cause "crow's feet" (eye narrowing)
        left_ear = self.get_ear(left_eye)
        right_ear = self.get_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        # Genuine smile: eyes slightly narrowed + wide mouth + raised corners
        eyes_smiling = avg_ear < (self.baseline_ear * 0.9)  # Eyes narrow when smiling
        mouth_smiling = mouth_ratio > 3.0 and corners_raised
        
        # Real smile requires BOTH
        is_genuine_smile = eyes_smiling and mouth_smiling
        
        return mouth_ratio, is_genuine_smile
    
    def detect_anger(self, landmarks, mouth):
        """Detect anger indicators - NEW FUNCTION"""
        # 1. Eyebrow lowering and furrowing (key anger indicator)
        left_brow_inner = landmarks[21]
        right_brow_inner = landmarks[22]
        left_eye_top = landmarks[37]
        right_eye_top = landmarks[44]
        nose_bridge = landmarks[27]
        
        # Distance between inner eyebrows (furrowing)
        brow_furrow_dist = dist.euclidean(left_brow_inner, right_brow_inner)
        
        # Brow position relative to eyes (lowered brows = anger)
        left_brow_dist = dist.euclidean(left_brow_inner, left_eye_top)
        right_brow_dist = dist.euclidean(right_brow_inner, right_eye_top)
        avg_brow_dist = (left_brow_dist + right_brow_dist) / 2
        
        # 2. Lip compression (pressed lips = anger)
        upper_lip_inner = landmarks[62]
        lower_lip_inner = landmarks[66]
        lip_compression = dist.euclidean(upper_lip_inner, lower_lip_inner)
        
        # 3. Mouth tension (straight, tense mouth)
        mouth_corners_y_diff = abs(mouth[0][1] - mouth[6][1])
        
        # 4. Jaw tension
        left_jaw = landmarks[4]
        right_jaw = landmarks[12]
        chin = landmarks[8]
        jaw_width = dist.euclidean(left_jaw, right_jaw)
        face_height = dist.euclidean(nose_bridge, chin)
        jaw_ratio = jaw_width / (face_height + 1e-6)
        
        # Anger indicators
        anger_score = 0
        anger_indicators = []
        
        # Lowered, furrowed brows (STRONGEST anger indicator)
        if avg_brow_dist < (self.baseline_brow_dist * 0.85):
            anger_score += 30
            anger_indicators.append('lowered_brows')
        
        if brow_furrow_dist < 35:  # Close together = furrowed
            anger_score += 25
            anger_indicators.append('furrowed_brows')
        
        # Lip compression
        if lip_compression < 5:
            anger_score += 20
            anger_indicators.append('lip_compression')
        
        # Jaw tension
        if jaw_ratio > 0.85:
            anger_score += 15
            anger_indicators.append('jaw_tension')
        
        # Mouth corners level (not raised like smile)
        if mouth_corners_y_diff < 3:
            anger_score += 10
            anger_indicators.append('tense_mouth')
        
        is_angry = anger_score > 50
        
        return anger_score, is_angry, anger_indicators
    
    def detect_eyebrow_raise(self, landmarks):
        """Detect raised eyebrows (stress/surprise indicator)"""
        left_brow = landmarks[19]
        right_brow = landmarks[24]
        left_eye_top = landmarks[37]
        right_eye_top = landmarks[44]
        
        left_dist = dist.euclidean(left_brow, left_eye_top)
        right_dist = dist.euclidean(right_brow, right_eye_top)
        
        avg_brow_dist = (left_dist + right_dist) / 2
        return avg_brow_dist
    
    def detect_jaw_clench(self, face_landmarks):
        """Detect jaw clenching (stress indicator)"""
        left_jaw = face_landmarks[4]
        right_jaw = face_landmarks[12]
        chin = face_landmarks[8]
        nose_bridge = face_landmarks[27]
        
        jaw_width = dist.euclidean(left_jaw, right_jaw)
        face_height = dist.euclidean(nose_bridge, chin)
        
        jaw_ratio = jaw_width / (face_height + 1e-6)
        return jaw_ratio
    
    def analyze_frame(self, frame):
        """Analyze a single frame with improved stress and anger detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Extract features
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]
        
        # Calculate basic metrics
        ear = (self.get_ear(left_eye) + self.get_ear(right_eye)) / 2.0
        mar = self.get_mar(mouth)
        smile_ratio, is_smiling = self.detect_smile(mouth, left_eye, right_eye)
        brow_height = self.detect_eyebrow_raise(landmarks)
        jaw_ratio = self.detect_jaw_clench(landmarks)
        
        # NEW: Detect anger
        anger_score, is_angry, anger_indicators = self.detect_anger(landmarks, mouth)
        
        # Detect blinks
        self.ear_history.append(ear)
        if len(self.ear_history) >= self.BLINK_THRESHOLD:
            recent_ears = list(self.ear_history)[-self.BLINK_THRESHOLD:]
            if all(e < self.EAR_THRESHOLD for e in recent_ears):
                if self.blink_frames == 0:
                    current_time = time.time()
                    if current_time - self.last_blink_time < 2.0:
                        self.rapid_blink_count += 1
                    self.last_blink_time = current_time
                    self.blink_count += 1
                    self.blink_frames = self.BLINK_THRESHOLD
            else:
                self.blink_frames = max(0, self.blink_frames - 1)
        
        # Calibration phase
        if self.calibration_frames < self.calibration_needed:
            self.baseline_ear = (self.baseline_ear * self.calibration_frames + ear) / (self.calibration_frames + 1)
            self.baseline_mar = (self.baseline_mar * self.calibration_frames + mar) / (self.calibration_frames + 1)
            self.baseline_smile = (self.baseline_smile * self.calibration_frames + smile_ratio) / (self.calibration_frames + 1)
            self.baseline_brow_dist = (self.baseline_brow_dist * self.calibration_frames + brow_height) / (self.calibration_frames + 1)
            self.calibration_frames += 1
            stress_score = 0
            stress_level = "Calibrating"
        else:
            self.calibrated = True
            
            # IMPROVED STRESS CALCULATION with ANGER detection
            stress_indicators = []
            
            # CRITICAL: If angry, high stress regardless of other factors
            if is_angry:
                stress_indicators.append(('anger', anger_score))
            
            # 1. Eye strain (deviation from baseline EAR)
            if not is_smiling:  # Don't count if genuinely smiling
                ear_deviation = abs(ear - self.baseline_ear)
                if ear_deviation > 0.05:
                    stress_indicators.append(('eye_strain', ear_deviation * 20))
            
            # 2. Mouth tension (but NOT if genuinely smiling)
            if not is_smiling:
                mar_deviation = abs(mar - self.baseline_mar)
                if mar_deviation > 0.1:
                    stress_indicators.append(('mouth_tension', mar_deviation * 15))
            
            # 3. Rapid blinking (stress indicator)
            if self.rapid_blink_count > 3:
                stress_indicators.append(('rapid_blink', min(self.rapid_blink_count * 5, 30)))
            
            # 4. Low blink rate (concentration/stress)
            blink_rate = self.blink_count / (self.calibration_frames / 30.0)
            if blink_rate < 0.15:
                stress_indicators.append(('low_blink', 15))
            
            # 5. Raised eyebrows (surprise/stress) - but not if angry
            if not is_angry and brow_height > (self.baseline_brow_dist * 1.15):
                stress_indicators.append(('raised_brows', 10))
            
            # Calculate final stress score
            stress_score = sum(score for _, score in stress_indicators)
            stress_score = min(stress_score, 100)  # Cap at 100
            
            # Smooth stress values
            self.stress_history.append(stress_score)
            smoothed_stress = np.mean(list(self.stress_history)[-30:]) if self.stress_history else stress_score
            
            # Determine stress level
            if smoothed_stress > 60:
                stress_level = "High"
            elif smoothed_stress > 30:
                stress_level = "Moderate"
            else:
                stress_level = "Low"
            
            # Reset rapid blink counter periodically
            if time.time() - self.last_blink_time > 5.0:
                self.rapid_blink_count = 0
        
        return {
            'ear': ear,
            'mar': mar,
            'smile_ratio': smile_ratio,
            'is_smiling': is_smiling,
            'is_angry': is_angry,
            'anger_score': anger_score,
            'anger_indicators': anger_indicators,
            'stress_score': stress_score if self.calibrated else 0,
            'stress_level': stress_level,
            'blinks': self.blink_count,
            'rapid_blinks': self.rapid_blink_count,
            'calibrated': self.calibrated,
            'landmarks': landmarks
        }
    
def run_improved_detection(duration_seconds=20, test_mode=False, show_display=True):
    """Run improved stress detection"""
    
    print("\n" + "="*70)
    print(" IMPROVED VIDEO STRESS DETECTION (Smile & Anger Aware)")
    if test_mode:
        print(" TEST MODE: Press 's' for stressed, 'n' for normal")
    print("="*70)
    
    detector = ImprovedStressDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Camera not accessible!")
        return 0.0, "Unknown"
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n Recording for {duration_seconds} seconds...")
    print(" Look at the camera naturally")
    print("😊 Try smiling - it should NOT increase stress!")
    print("😠 Try looking angry - it SHOULD increase stress!")
    if test_mode:
        print("  Press 's' for stressed, 'n' for normal during recording")
    print()
    
    stress_values = []
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    while (time.time() - start_time) < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Analyze frame
        result = detector.analyze_frame(frame)
        
        if result:
            stress_values.append(result['stress_score'])
            
            # Display with landmarks
            if show_display:
                try:
                    # Draw landmarks
                    for (x, y) in result['landmarks']:
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    
                    # Draw info
                    h, w = frame.shape[:2]
                    
                    # Status
                    status = " Calibrating..." if not result['calibrated'] else "✅ Detecting"
                    cv2.putText(frame, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Emotion indicators
                    y_pos = 60
                    if result['is_smiling']:
                        cv2.putText(frame, "😊 GENUINE SMILE", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        y_pos += 30
                    
                    if result['is_angry']:
                        anger_text = f"😠 ANGER DETECTED ({result['anger_score']:.0f})"
                        cv2.putText(frame, anger_text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        y_pos += 30
                        # Show which indicators triggered
                        if result['anger_indicators']:
                            indicators_text = ", ".join(result['anger_indicators'][:2])
                            cv2.putText(frame, f"  ({indicators_text})", (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            y_pos += 25
                    
                    # Stress level
                    stress_color = (0, 255, 0) if result['stress_level'] == "Low" else \
                                   (0, 165, 255) if result['stress_level'] == "Moderate" else (0, 0, 255)
                    cv2.putText(frame, f"Stress: {result['stress_level']} ({result['stress_score']:.0f})", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2)
                    y_pos += 30
                    
                    # Blinks
                    cv2.putText(frame, f"Blinks: {result['blinks']}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Instructions
                    if test_mode:
                        cv2.putText(frame, "Press 's' = stressed | 'n' = normal | 'q' = quit", 
                                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, "Press 'q' to quit", 
                                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow("Improved Stress Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif test_mode and key == ord('s'):
                        prediction = 1 if result['stress_score'] > 40 else 0
                        detector.ground_truth.append(1)
                        detector.predictions.append(prediction)
                        print(f"  ✓ Labeled as STRESSED (prediction: {'stressed' if prediction == 1 else 'normal'})")
                    elif test_mode and key == ord('n'):
                        prediction = 1 if result['stress_score'] > 40 else 0
                        detector.ground_truth.append(0)
                        detector.predictions.append(prediction)
                        print(f"  ✓ Labeled as NORMAL (prediction: {'stressed' if prediction == 1 else 'normal'})")
                
                except cv2.error:
                    show_display = False
                    print("  GUI not available, continuing without display...")
            
            # Print status every 2 seconds
            if time.time() - last_print_time > 2:
                status = " Calibrating..." if not result['calibrated'] else "✅ Detecting"
                smile_ind = "😊" if result['is_smiling'] else "  "
                anger_ind = "😠" if result['is_angry'] else "  "
                print(f"{status} {smile_ind}{anger_ind} | Stress: {result['stress_level']:8s} ({result['stress_score']:3.0f}) | "
                      f"Blinks: {result['blinks']:2d}")
                last_print_time = time.time()
    
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Calculate results
    avg_stress = np.mean(stress_values) if stress_values else 0
    stress_level = "High" if avg_stress > 60 else "Moderate" if avg_stress > 30 else "Low"
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total blinks detected:  {detector.blink_count}")
    print(f"  Rapid blinks detected:  {detector.rapid_blink_count}")
    print(f"  Average stress score:   {avg_stress:.2f}/100")
    print(f"  Stress level:           {stress_level}")
    print(f"  Status:                 {' CALIBRATED' if detector.calibrated else '⚠️ NOT CALIBRATED'}")
    
    # Show accuracy metrics
    # if test_mode:
    #     metrics = detector.calculate_accuracy()
    #     if metrics:
    #         print("\n" + "="*70)
    #         print(" ACCURACY METRICS:")
    #         print("="*70)
    #         print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    #         print(f"  Precision: {metrics['precision']:.2%}")
    #         print(f"  Recall:    {metrics['recall']:.2%}")
    #         print(f"  F1-Score:  {metrics['f1_score']:.2%}")
            
    #         print("\nCONFUSION MATRIX:")
    #         cm = metrics['confusion_matrix']
    #         print(f"                Predicted")
    #         print(f"                Normal  Stressed")
    #         print(f"  Actual Normal    {cm[0][0]:3d}     {cm[0][1]:3d}")
    #         print(f"  Actual Stressed  {cm[1][0]:3d}     {cm[1][1]:3d}")
            
    #         print("\n CLASSIFICATION REPORT:")
    #         print(metrics['classification_report'])
    #         print(f"  Total labeled samples: {len(detector.ground_truth)}")
    #     else:
    #         print("\n  No ground truth labels provided")
    
    print("="*70 + "\n")
    
    return avg_stress / 100, stress_level


if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    duration = 45
    
    for arg in sys.argv[1:]:
        if arg.isdigit():
            duration = int(arg)
    
    try:
        run_improved_detection(duration_seconds=duration, test_mode=test_mode)
    except KeyboardInterrupt:
        print("\n Stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()