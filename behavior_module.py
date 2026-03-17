import cv2
import mediapipe as mp
import numpy as np

class BlinkState:
    OPEN = 0
    CLOSED = 1

class BehaviorAnalyzer:
    def __init__(self):
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        # Threshold: Isse niche EAR jayega toh aankh "Band" mani jayegi
        self.EAR_THRESHOLD = 0.23 
        
        # Automata Variables
        self.state = BlinkState.OPEN  
        self.blink_count = 0          
        
        # --- NEW: Slow Blink aur Noise control ke liye ---
        self.cooldown_frames = 0 
        self.MIN_COOLDOWN = 15  # Ek blink ke baad 15 frames tak doosra count lock rahega

    def _calculate_ear(self, landmarks, eye_indices):
        """Eye Aspect Ratio (EAR) nikalne ka formula"""
        pts = [np.array(landmarks[i]) for i in eye_indices]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_behavior(self, frame, landmarks, frame_count):
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        l_ear = self._calculate_ear(landmarks, LEFT_EYE)
        r_ear = self._calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (l_ear + r_ear) / 2.0

        # --- REFINED AUTOMATA LOGIC ---
        
        # 1. Cooldown timer ko kam karte raho (agar chal raha hai)
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        # 2. Transition: OPEN se CLOSED (Aankh band ho rahi hai)
        if self.state == BlinkState.OPEN and avg_ear < self.EAR_THRESHOLD:
            # Agar cooldown zero hai tabhi 'CLOSED' state mein jao
            if self.cooldown_frames == 0:
                self.state = BlinkState.CLOSED
            
        # 3. Transition: CLOSED se OPEN (Aankh wapas khul gayi)
        elif self.state == BlinkState.CLOSED and avg_ear > self.EAR_THRESHOLD:
            self.blink_count += 1       # Aik blink count karo
            self.state = BlinkState.OPEN  # Wapas OPEN state
            self.cooldown_frames = self.MIN_COOLDOWN # Agle 15 frames ke liye lock laga do

        # Sirf Blinking Count return karna
        return f"Blinks Counted: {self.blink_count}"

    def get_final_report(self):
        return {"total_blinks": self.blink_count}