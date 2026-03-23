import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend

def apply_bandpass(signal, fs=30):
    # Strictly Human: 0.83Hz to 2.0Hz (Approx 50 to 120 BPM)
    # Khalid Mehmood Rule: 140+ is noise for a sitting person.
    low = 0.83 / (0.5 * fs)
    high = 2.0 / (0.5 * fs)
    b, a = butter(2, [low, high], btype='band')
    
    if len(signal) > 10:
        signal = detrend(signal) # Movement artifacts hatane ke liye
    
    return filtfilt(b, a, signal)

def apply_fft(signal, fs):
    n = len(signal)
    if n < fs * 5: # Minimum 5 seconds for stability
        return 0, 0

    filtered = apply_bandpass(signal, fs)
    
    # FFT Calculation
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # Valid human heart range (50 - 120 BPM)
    valid_idx = np.where((freqs >= 0.83) & (freqs <= 2.0))[0]

    if len(valid_idx) > 0:
        # Peak Detection
        best_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
        bpm = freqs[best_idx] * 60
        
        # SNR Logic: Peak power vs background noise
        peak_power = fft_vals[best_idx]
        total_power = np.sum(fft_vals[valid_idx])
        snr = peak_power / (total_power - peak_power + 1e-6)
        
        # Khalid Mehmood Consistency Check:
        # Agar SNR 0.2 se kam hai, matlab signal mein periodicity nahi hai (Deepfake sign)
        confidence = snr * 10 
        
        # Strict BPM Cap: Noise rejection
        if bpm > 130 or bpm < 45:
            return 0, 0
            
        return bpm, confidence

    return 0, 0

def extract_skin_green(frame, landmarks):
    h, w, _ = frame.shape
    # Landmark 10: Forehead center (Best for rPPG)
    lm = landmarks.landmark[10]
    x, y = int(lm.x * w), int(lm.y * h)

    # ROI for forehead
    r = int(min(h, w) * 0.035)
    roi = frame[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)]

    if roi.size == 0: return None, (x, y)

    # YCrCb is superior for skin detection under varying light
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    green_channel = roi[:,:,1]
    green_skin = green_channel[mask > 0]

    if len(green_skin) > 20:
        # Green channel has highest hemoglobin absorption info
        return np.mean(green_skin), (x, y)
    
    return np.mean(green_channel), (x, y)