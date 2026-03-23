import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import rppg_module
import behavior_module

# --- CONFIG ---
BUFFER_SIZE = 150 
green_signal = []
smoothed_bpm = 0  

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# --- CLEAN INPUT FLOW ---
print("\n" + "="*40)
print("  DEEPFAKE DETECTION SYSTEM (V3.2 - K.MEHMOOD LOGIC)")
print("="*40)
print("1 : Webcam")
print("2 : Video File")

mode = input(">> Select Mode (1 or 2): ").strip()

cap = None
if mode == "1":
    cap = cv2.VideoCapture(0)
else:
    file_name = input("\n[?] Enter video filename: ").strip()
    full_path = os.path.join("videos", file_name) 
    cap = cv2.VideoCapture(full_path)

if not cap or not cap.isOpened():
    print("Error: Video source load nahi ho saki.")
    exit()

analyzer = behavior_module.BehaviorAnalyzer()
frame_id = 0
FS = cap.get(cv2.CAP_PROP_FPS)
if FS <= 0 or FS is None: FS = 30  

# --- THE CROSS-VERIFICATION BRAIN ---
def get_final_decision(rppg_status, bhv_status, bhv_score, rppg_conf, bpm):
    if rppg_status == "WAITING" and bhv_score > 90:
        return "ANALYZING...", (255, 255, 255)

    # 1. Khalid Mehmood's Physiological Rejector
    # Insaan ka 45-55 BPM static video mein hona deepfake artifacts ho sakte hain
    is_bpm_suspicious = (bpm < 56 or bpm > 115) and rppg_conf > 1.0

    # 2. Strict Behavioral Penalty
    # Agar Unnatural blinks zyada hain toh score gir chuka hoga
    is_behavior_fake = (bhv_status == "FAKE" or bhv_score < 45)

    # --- FINAL VERDICT LOGIC ---
    if is_behavior_fake:
        return "HIGH RISK (FAKE BEHAVIOR)", (0, 0, 255) # RED
    
    if is_bpm_suspicious:
        # Agar blinking theek hai magar HR weird hai
        if bhv_score > 85:
            return "SUSPICIOUS (ABNORMAL HR)", (0, 165, 255) # ORANGE
        return "HIGH RISK (FAKE HR)", (0, 0, 255) # RED

    if rppg_status == "REAL" and bhv_status == "REAL" and bhv_score >= 75:
        return "SECURE (REAL)", (0, 255, 0) # GREEN

    return "SUSPICIOUS", (0, 165, 255)

# ----------- DASHBOARD & GRAPH FUNCTIONS RE-OPTIMIZED -----------
def draw_graph(frame, data):
    h, w, _ = frame.shape
    graph_h, graph_w = 110, 320
    graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
    if len(data) > 1:
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            norm = ((data - d_min) / (d_max - d_min + 1e-6) * (graph_h - 20)).astype(int)
            for i in range(1, len(norm)):
                cv2.line(graph, (int((i-1)*graph_w/len(norm)), graph_h-10-norm[i-1]), 
                         (int(i*graph_w/len(norm)), graph_h-10-norm[i]), (0, 255, 0), 1)
    frame[h-graph_h-10 : h-10, 10 : 10+graph_w] = graph

# ---------------- MAIN LOOP ----------------
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_id += 1
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    rppg_status, rppg_conf = "WAITING", 0.0
    bhv_status, bhv_score, bhv_text = "REAL", 100, "Analyzing..."
    bpm_display = "Collecting Data..."

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 1. Physiological (rPPG)
            avg_green, pos = rppg_module.extract_skin_green(frame, landmarks)
            if avg_green is not None:
                green_signal.append(avg_green)
                if len(green_signal) > BUFFER_SIZE: green_signal.pop(0)
                if len(green_signal) == BUFFER_SIZE:
                    bpm, rppg_conf = rppg_module.apply_fft(np.array(green_signal), fs=FS)
                    if 45 < bpm < 140:
                        smoothed_bpm = bpm if smoothed_bpm == 0 else (0.15 * bpm + 0.85 * smoothed_bpm)
                    # Strict validation for status
                    rppg_status = "REAL" if (56 < smoothed_bpm < 115 and rppg_conf > 1.4) else "FAKE"
                    bpm_display = f"{smoothed_bpm:.0f} BPM (Conf: {rppg_conf:.1f})"

            # 2. Behavioral (Blinking)
            lms = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
            bhv_text, bhv_status, bhv_score = analyzer.detect_behavior(frame, lms)
            
            # 3. Final Integration
            final_label, label_color = get_final_decision(rppg_status, bhv_status, bhv_score, rppg_conf, smoothed_bpm)

    # DASHBOARD UI
    cv2.rectangle(frame, (w-400, 20), (w-20, 180), (0,0,0), -1)
    cv2.putText(frame, f"SYSTEM: {final_label}", (w-385, 55), 1, 1.4, label_color, 2)
    cv2.putText(frame, f"Trust Score: {bhv_score}%", (w-385, 90), 1, 1.1, (255,255,255), 1)
    cv2.putText(frame, f"Heart Rate: {bpm_display}", (w-385, 125), 1, 1.1, (0,255,255), 1)
    cv2.putText(frame, bhv_text, (w-385, 160), 1, 0.9, (180,180,180), 1)

    draw_graph(frame, np.array(green_signal))
    cv2.imshow("Deepfake Detector V3.2", frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()