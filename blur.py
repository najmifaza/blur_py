import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Siapkan opsi untuk deteksi tangan
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

detector = vision.HandLandmarker.create_from_options(options)

# Daftar koneksi antar titik jari untuk menggambar garis
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8), # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12), # Tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Manis
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Kelingking
]

# Fungsi hitung jarak antar 2 titik
def get_distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

# Fungsi deteksi jari terangkat (berdasarkan jarak ke pergelangan tangan)
def finger_up(tip, pip, landmarks):
    # Jika jarak ujung jari ke pergelangan (0) LEBIH JAUH daripada sendi tengah (pip) ke pergelangan,
    # maka jari tersebut sedang lurus/terangkat. (Tidak peduli tangan miring atau terbalik!)
    dist_tip = get_distance(landmarks[tip], landmarks[0])
    dist_pip = get_distance(landmarks[pip], landmarks[0])
    return dist_tip > dist_pip

def count_fingers_up(landmarks):
    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)
    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    return fingers.count(True)

# open camera
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Konversi frame ke format gambar MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Lakukan deteksi tangan
    hand_result = detector.detect(mp_image)

    peace_detected = False
    fingers_count = 0

    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            fingers_count = count_fingers_up(hand_landmarks)
            if fingers_count == 2:
                peace_detected = True
                break

    # blur efek
    if peace_detected:
        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )

    # Gambar kerangka tangan
    if hand_result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in hand_result.hand_landmarks:
            # Ubah koordinat relatif ke pixel
            pixel_landmarks = []
            for landmark in hand_landmarks:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                pixel_landmarks.append((px, py))
            
            # Gambar garis koneksi antar titik (hijau)
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                    cv2.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)
            
            # Gambar titik jari (merah)
            for px, py in pixel_landmarks:
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    cv2.imshow(
        "Peace Blur",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
