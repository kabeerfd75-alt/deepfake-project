import os
# TensorFlow/MediaPipe ki faltu warnings chupane ke liye
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import rppg_module
import behavior_module

# --- CONFIG ---
BUFFER_SIZE = 300
FS = 30
green_signal = []

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# --- CLEAN INPUT FLOW ---
print("\n" + "="*30)
print("   DEEPFAKE DETECTION SYSTEM")
print("="*30)
print("1 : Webcam")
print("2 : Video File")

mode = input(">> Select Mode (1 or 2): ").strip()

cap = None
if mode == "1":
    cap = cv2.VideoCapture(0)
    print("\n[+] Initializing Webcam...")
else:
    path = input("\n[?] Enter video path: ").strip().replace('"', '').replace("'", "")
    cap = cv2.VideoCapture(path)
    print(f"\n[+] Loading video: {path}")

if not cap or not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("System running... Press ESC to exit")

analyzer = behavior_module.BehaviorAnalyzer()
frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 1. rPPG Extraction
            avg_green, pos = rppg_module.extract_skin_green(frame, landmarks)
            x, y = pos
            r = 30
            cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0, 255, 0), 2)
            if avg_green is not None:
                green_signal.append(avg_green)
                if len(green_signal) > BUFFER_SIZE: green_signal.pop(0)

                # 2. Signal Processing & Detection
                if len(green_signal) == BUFFER_SIZE:
                    filtered = rppg_module.bandpass_filter(np.array(green_signal))
                    bpm, peak = rppg_module.apply_fft(filtered)
                    periodicity = rppg_module.check_periodicity(filtered)
                    
                    status = "REAL" if (bpm > 40 and bpm < 180) else "FAKE"
                    
                    # UI Rendering
                    cv2.putText(frame, f"BPM: {bpm:.0f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Status: {status}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- 3. ASAD'S BEHAVIOR MODULE (REPLACED) ---
            # Step A: Landmarks ko normalized se pixel coordinates mein convert karein
            h, w, _ = frame.shape
            landmarks_list = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

            # Step B: Behavior detect karein (Class method call)
            # Note: frame_id loop ke start mein 'frame_id += 1' karke initialize kar lena
            behavior_status = analyzer.detect_behavior(frame, landmarks_list, frame_id)
            
            # Step C: UI Rendering with Dynamic Color
            # Agar Suspicious ho toh Red, warna Cyan
            color = (0, 0, 255) if "Suspicious" in behavior_status else (255, 255, 0)
            cv2.putText(frame, f"Behavior: {behavior_status}", (20, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. Draw Graph (Always)
    rppg_module.draw_graph(frame, np.array(green_signal))
    
    cv2.imshow("Deepfake Detector System", frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()