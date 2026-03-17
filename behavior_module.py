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
        self.state = BlinkState.OPEN  # Shuruati state: Aankh khuli hai
        self.blink_count = 0          # Total blinks store karne ke liye

    def _calculate_ear(self, landmarks, eye_indices):
        """Eye Aspect Ratio (EAR) nikalne ka formula"""
        pts = [np.array(landmarks[i]) for i in eye_indices]
        # Vertical distances (Aankh ki unchai)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance (Aankh ki chorai)
        h = np.linalg.norm(pts[0] - pts[3])
        # EAR Formula
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_behavior(self, frame, landmarks, frame_count):
        """
        Main function jo blinking count karta hai
        """
        # MediaPipe ke landmark indices (Left aur Right Eye)
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Dono aankhon ka EAR nikal kar average lena
        l_ear = self._calculate_ear(landmarks, LEFT_EYE)
        r_ear = self._calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (l_ear + r_ear) / 2.0

        # --- AUTOMATA LOGIC START ---
        
        # Agar aankh khuli hai aur EAR threshold se niche gir jaye (Aankh band ho rahi hai)
        if self.state == BlinkState.OPEN and avg_ear < self.EAR_THRESHOLD:
            self.state = BlinkState.CLOSED # State badal kar 'CLOSED' kardo
            
        # Agar aankh band hai aur EAR threshold se upar aa jaye (Aankh wapas khul gayi)
        elif self.state == BlinkState.CLOSED and avg_ear > self.EAR_THRESHOLD:
            self.blink_count += 1       # Aik blink mukammal hui, count barhao
            self.state = BlinkState.OPEN  # State wapas 'OPEN' kardo
            
        # --- AUTOMATA LOGIC END ---

        # Sirf Blinking Count return karna
        return f"Blinks Counted: {self.blink_count}"

    def get_final_report(self):
        """Video ke aakhir mein count hasil karne ke liye"""
        return {"total_blinks": self.blink_count}