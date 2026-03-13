import cv2
from rppg_module import get_heart_rate
from behavior_module import get_blinks

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Logic integration
    bpm = get_heart_rate(frame)
    blinks = get_blinks(frame)

    # Screen par dikhayein
    cv2.putText(frame, f"BPM: {bpm}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blinks}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Deepfake Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()