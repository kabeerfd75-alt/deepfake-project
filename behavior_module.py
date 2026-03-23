import cv2
import mediapipe as mp
import numpy as np
import time

class BlinkState:
    OPEN = 0
    CLOSED = 1

class BehaviorAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        # --- KHALID MEHMOOD LOGIC THRESHOLDS ---
        self.EAR_THRESHOLD = 0.20  # Optimized for webcam noise
        self.MIN_BLINK_FRAMES = 2  # Human fast blink
        self.MAX_BLINK_FRAMES = 8  # Normal human blink duration limit
        
        self.state = BlinkState.OPEN  
        self.blink_count = 0 
        self.unnatural_blink_count = 0
        self.closed_frames_count = 0
        
        self.start_time = time.time()
        self.last_blink_time = time.time()
        self.MAX_NO_BLINK_INTERVAL = 15 # 20s was too long for security

        # --- DYNAMIC TRUST ENGINE ---
        self.current_trust_score = 100

    def _calculate_ear(self, landmarks, eye_indices):
        pts = [np.array(landmarks[i]) for i in eye_indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h)

    def detect_behavior(self, frame, landmarks):
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        l_ear = self._calculate_ear(landmarks, LEFT_EYE)
        r_ear = self._calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (l_ear + r_ear) / 2.0

        current_time = time.time()
        time_since_last = current_time - self.last_blink_time

        # --- REFINED STATE MACHINE ---
        if self.state == BlinkState.OPEN and avg_ear < self.EAR_THRESHOLD:
            self.state = BlinkState.CLOSED
            self.closed_frames_count = 0 

        elif self.state == BlinkState.CLOSED:
            if avg_ear < self.EAR_THRESHOLD:
                self.closed_frames_count += 1
            else:
                # BLINK COMPLETED: Validate Quality
                if self.MIN_BLINK_FRAMES <= self.closed_frames_count <= self.MAX_BLINK_FRAMES:
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    # RECOVERY: Increase score based on consistency
                    self.current_trust_score += 10 
                else:
                    self.unnatural_blink_count += 1
                    # PENALTY: Heavy reduction for robotic movement
                    self.current_trust_score -= 25
                
                self.state = BlinkState.OPEN

        # --- RATIO-BASED ADJUSTMENT (Professor Malik's Accuracy Fix) ---
        # Agar Natural Blinks > Unnatural hain, toh score ko girne se rokna hai
        total = self.blink_count + self.unnatural_blink_count
        if total > 5:
            unnatural_ratio = self.unnatural_blink_count / total
            if unnatural_ratio > 0.3: # 30% se zyada kachra blinks
                self.current_trust_score -= 2 # Constant drain

        # No Blink for long time
        if time_since_last > self.MAX_NO_BLINK_INTERVAL:
            self.current_trust_score -= 0.5

        # Final score clamping
        self.current_trust_score = max(0, min(100, self.current_trust_score))

        # Status Logic
        if self.current_trust_score < 45:
            behavior_status = "FAKE"
        elif self.current_trust_score < 75:
            behavior_status = "SUSPICIOUS"
        else:
            behavior_status = "REAL"

        output_text = f"Blinks: {self.blink_count} (Unnatural: {self.unnatural_blink_count})"
        return output_text, behavior_status, int(self.current_trust_score)