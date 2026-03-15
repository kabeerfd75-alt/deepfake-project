import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt

# ---------------- SIGNAL PROCESSING ----------------

def bandpass_filter(signal, low=0.7, high=4, fs=30):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(1, [low, high], btype='band')
    return filtfilt(b, a, signal)

def apply_fft(signal, fs=30):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))

    peak_index = np.argmax(fft_vals)
    peak_freq = freqs[peak_index]

    bpm = peak_freq * 60
    peak_strength = fft_vals[peak_index]

    return bpm, peak_strength

def check_periodicity(signal, fs=30):
    signal = signal - np.mean(signal)
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[corr.size//2:]
    peak = np.max(corr[fs//2:])  # ignore lag=0
    normalized = peak / np.sum(signal**2 + 1e-6)
    return normalized

# ---------------- ROI EXTRACTION ----------------

def extract_skin_green(frame, landmarks):
    h, w, _ = frame.shape
    # Forehead center
    lm = landmarks.landmark[10]
    x = int(lm.x * w)
    y = int(lm.y * h)
    r = 30
    roi = frame[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)]

    # HSV skin mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0,30,60])
    upper = np.array([20,150,255])
    mask = cv2.inRange(hsv, lower, upper)

    green = roi[:,:,1]
    green_skin = green[mask>0]

    if len(green_skin) == 0:
        return None, (x,y)
    return np.mean(green_skin), (x,y)

# ---------------- GRAPH DRAWING ----------------

def draw_graph(frame, data):
    graph_height = 120
    graph_width = 300
    graph = np.zeros((graph_height, graph_width,3), dtype=np.uint8)

    if len(data) > 1:
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
        normalized = (normalized * (graph_height-10)).astype(int)

        for i in range(1,len(normalized)):
            x1 = int((i-1)/len(normalized)*graph_width)
            x2 = int(i/len(normalized)*graph_width)
            y1 = graph_height - normalized[i-1]
            y2 = graph_height - normalized[i]
            cv2.line(graph,(x1,y1),(x2,y2),(0,255,0),1)

    frame[10:10+graph_height, 10:10+graph_width] = graph

# ---------------- DEEPFAKE HEURISTIC ----------------

def detect_deepfake_robust(peak_strength, bpm, periodicity):
    if bpm < 40 or bpm > 180:
        return "FAKE"
    if peak_strength < 5 or periodicity < 0.3:
        return "POSSIBLE FAKE"
    return "REAL"

# ---------------- VIDEO SOURCE ----------------

print("Select Mode:")
print("1 : Webcam")
print("2 : Video File")
mode = input("Enter choice: ")

if mode == "1":
    cap = cv2.VideoCapture(0)
    print("Video opened:", cap.isOpened())
else:
    path = input("Enter video path: ")
    cap = cv2.VideoCapture(path)

# ---------------- MEDIAPIPE SETUP ----------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- PARAMETERS ----------------

green_signal = []
buffer_size = 300
fs = 30

print("System running... Press ESC to exit")

# ---------------- MAIN LOOP ----------------

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            avg_green, pos = extract_skin_green(frame, landmarks)
            if avg_green is not None:
                green_signal.append(avg_green)
                if len(green_signal) > buffer_size:
                    green_signal.pop(0)

                x,y = pos
                r = 30
                cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0,255,0), 2)
                cv2.putText(frame,f"Signal: {avg_green:.2f}",
                            (20,180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),2)

                if len(green_signal) == buffer_size:
                    filtered = bandpass_filter(np.array(green_signal), fs=fs)
                    bpm, peak = apply_fft(filtered, fs)
                    periodicity = check_periodicity(filtered, fs)

                    status = detect_deepfake_robust(peak, bpm, periodicity)

                    cv2.putText(frame,
                                f"Heart Rate: {bpm:.0f} BPM",
                                (20,210),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,255,0),
                                2)
                    cv2.putText(frame,
                                f"Status: {status}",
                                (20,240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0,0,255),
                                2)

                draw_graph(frame, np.array(green_signal))

    cv2.imshow("rPPG Deepfake Detector", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()