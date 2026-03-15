import cv2
import numpy as np
from scipy.signal import butter, filtfilt

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
    return freqs[peak_index] * 60, fft_vals[peak_index]

def check_periodicity(signal, fs=30):
    signal = signal - np.mean(signal)
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[corr.size//2:]
    peak = np.max(corr[fs//2:])
    return peak / np.sum(signal**2 + 1e-6)

def extract_skin_green(frame, landmarks):
    h, w, _ = frame.shape
    # Forehead center
    lm = landmarks.landmark[10]
    x, y = int(lm.x * w), int(lm.y * h)
    r = 30
    roi = frame[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)]

    if roi.size == 0: return None, (x, y)

    # HSV skin mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,30,60]), np.array([20,150,255]))
    green = roi[:,:,1]
    green_skin = green[mask>0]

    return np.mean(green_skin) if len(green_skin) > 0 else None, (x, y)

def draw_graph(frame, data):
    graph_height, graph_width = 120, 300
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    if len(data) > 1:
        norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
        norm = (norm * (graph_height-10)).astype(int)
        for i in range(1, len(norm)):
            cv2.line(graph, (int((i-1)/len(norm)*graph_width), graph_height - norm[i-1]),
                            (int(i/len(norm)*graph_width), graph_height - norm[i]), (0, 255, 0), 1)
    frame[10:10+graph_height, 10:10+graph_width] = graph