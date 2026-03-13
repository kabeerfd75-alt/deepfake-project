import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, resample

# ----------------- Functions -----------------
def bandpass_filter(signal, low=0.7, high=4, fs=30):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_bpm(signal, fs=30):
    n = len(signal)
    if n < 2: 
        return 0
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.abs(np.fft.rfft(signal - np.mean(signal)))
    peak_idx = np.argmax(fft_values)
    peak_freq = freqs[peak_idx]
    bpm = peak_freq * 60
    return bpm

def draw_pulse_graph(frame, signal_buffer, width=200, height=80, x=20, y=150):
    if len(signal_buffer) < 2:
        return
    # normalize signal for plotting
    sig = np.array(signal_buffer[-200:])  # last 200 samples
    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
    sig = (sig * height).astype(int)
    for i in range(1, len(sig)):
        cv2.line(frame, (x + i - 1, y + height - sig[i - 1]),
                 (x + i, y + height - sig[i]), (0, 255, 0), 1)

# ----------------- Mediapipe Setup -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------- Webcam Setup -----------------
cap = cv2.VideoCapture(0)
print("System Start: Forehead rPPG scan ho raha hai...")

green_signal = []
buffer_size = 300  # ~10 sec buffer at 30fps
fs = 30

# ----------------- Main Loop -----------------
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            lm = face_landmarks.landmark[10]  # forehead center
            x, y = int(lm.x * w), int(lm.y * h)

            r = 20  # ROI size
            x1, y1 = max(x - r, 0), max(y - r, 0)
            x2, y2 = min(x + r, w), min(y + r, h)

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Draw rectangle ROI instead of dot
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            avg_green = np.mean(roi[:, :, 1])
            green_signal.append(avg_green)
            if len(green_signal) > buffer_size:
                green_signal.pop(0)

            # Signal quality check
            if avg_green < 30:  # very dark or low light
                cv2.putText(image, "Signal Quality Low", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, f"Signal: {avg_green:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Heart rate calculation
            if len(green_signal) == buffer_size:
                filtered_signal = bandpass_filter(np.array(green_signal), low=0.7, high=4, fs=fs)
                bpm = calculate_bpm(filtered_signal, fs=fs)
                cv2.putText(image, f"Heart Rate: {bpm:.0f} BPM", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw live pulse wave graph
            draw_pulse_graph(image, green_signal, width=200, height=80, x=20, y=120)

    else:
        cv2.putText(image, "Face Lost - Reposition Head", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Deepfake Tracker", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()