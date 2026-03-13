import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt

# Mediapipe Setup (Function ke bahar takay baar-baar init na ho)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
green_signal = []
buffer_size = 300 

def bandpass_filter(signal, low=0.7, high=4, fs=30):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_bpm(signal, fs=30):
    n = len(signal)
    if n < 2: return 0
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.abs(np.fft.rfft(signal - np.mean(signal)))
    peak_idx = np.argmax(fft_values)
    return freqs[peak_idx] * 60

def get_heart_rate(frame):
    global green_signal # taake purana data yaad rahe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    bpm = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            lm = face_landmarks.landmark[10]
            x, y = int(lm.x * w), int(lm.y * h)
            r = 20
            roi = frame[max(y-r, 0):min(y+r, h), max(x-r, 0):min(x+w, w)]
            
            if roi.size != 0:
                avg_green = np.mean(roi[:, :, 1])
                green_signal.append(avg_green)
                if len(green_signal) > buffer_size: green_signal.pop(0)

                if len(green_signal) == buffer_size:
                    filtered = bandpass_filter(np.array(green_signal))
                    bpm = calculate_bpm(filtered)
    
    return bpm # Yahan se value main.py mein jayegi